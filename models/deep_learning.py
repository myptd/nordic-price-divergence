"""
LSTM, TCN and Transformer for the 24 prices of D, used in the screening (models/screening_deep.py).

One example per delivery day D, with everything known at 11:00 on D-1:
  past block    168 hours, D-7 00:00 .. D-1 23:00: price (relative to the D-1 mean), load and wind/solar forecasts,
                hour and weekday encodings. All prices of D-1 are known (set by the D-2 auction).
  known for D   load and wind/solar forecasts for the 24 hours of D
  day level     weather at 10:00 on D-1, reservoir at 11:00 on D-1, commodities at D-2, calendar of D
                (features_day_ahead.day_level_features)
  output        24 values = price of D minus the D-1 mean (level-normalised target); the level is added back.

Network: encoder over the past block -> context vector; MLP head on [context, known-for-D, day level] -> 24 outputs,
plus a linear shortcut on [last 24 past prices, known-for-D, day level].
Training: AdamW, warm-up and cosine learning-rate decay, L1 loss, early stopping.
"""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import features_day_ahead as fd
import utils

MODELS = ["lstm", "tcn", "transformer"]
SEEDS = [42, 7, 123]
PAST_DAYS = 7

# Architecture and training (2 layers: one example per day, about 1,800-2,200 training days).
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.2
LR = {"lstm": 5e-4, "tcn": 5e-4, "transformer": 3e-4}
BATCH = 32
MAX_EPOCHS = 300
PATIENCE = 25
WARMUP_EPOCHS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALE = False   # level-only target; True would also divide price and target by S (std of D-1)


# ---------------------------------------------------------------------------
# Data: one example per delivery day
# ---------------------------------------------------------------------------

def build_arrays(hourly: pd.DataFrame) -> dict:
    """Past block (N, 168, Fp), known-for-D (N, 24, Fk), day level (N, S), target and mask (N, 24)."""
    utils.require_commodities(hourly.columns, "The neural networks")
    df = fd.prepare_hourly(hourly)
    P, mask = fd.daily_matrix(df[fd.TARGET])
    dates = P.index
    known_cols = ["load_forecast_mw"] + [c for c in df.columns if c.startswith("wsf_") and c != "wsf_total"]
    K = {c: fd.daily_matrix(df[c])[0] for c in known_cols}
    level = P.shift(1).mean(axis=1)
    spread = np.maximum(P.shift(1).std(axis=1), 1.0) if SCALE else pd.Series(1.0, index=dates)

    hour = np.arange(24)
    hsin, hcos = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    past = []
    for k in range(PAST_DAYS, 0, -1):                 # D-7 .. D-1, in time order
        day = dates - pd.Timedelta(days=k)
        feats = [(P.shift(k).values - level.values[:, None]) / spread.values[:, None]]
        feats += [K[c].shift(k).values for c in known_cols]
        feats += [np.broadcast_to(hsin, (len(dates), 24)), np.broadcast_to(hcos, (len(dates), 24))]
        feats += [np.repeat(np.sin(2 * np.pi * day.dayofweek.values / 7)[:, None], 24, 1),
                  np.repeat(np.cos(2 * np.pi * day.dayofweek.values / 7)[:, None], 24, 1)]
        past.append(np.stack(feats, axis=-1))         # (N, 24, Fp)
    past = np.concatenate(past, axis=1)               # (N, 168, Fp)
    future = np.stack([K[c].values for c in known_cols], axis=-1)    # (N, 24, Fk)
    static = fd.day_level_features(df, dates).drop(columns=["day_of_week", "month"])
    static["price_level"] = level
    static["price_d1_std"] = P.shift(1).std(axis=1)

    ok = (np.isfinite(past).all(axis=(1, 2)) & np.isfinite(future).all(axis=(1, 2))
          & static.notna().all(axis=1).values & P.notna().all(axis=1).values & level.notna().values)
    return {"dates": dates[ok], "past": past[ok], "future": future[ok], "static": static.values[ok].astype(float),
            "y": P.values[ok], "mask": mask.values[ok], "level": level.values[ok], "spread": spread.values[ok]}


class Scaler:
    """Mean/std per feature, fitted on the training days only."""

    def fit(self, a, axes):
        self.m, self.s = a.mean(axis=axes, keepdims=True), a.std(axis=axes, keepdims=True) + 1e-6
        return self

    def __call__(self, a):
        return (a - self.m) / self.s


def tensors(A, rows, sc):
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return (t(sc["past"](A["past"][rows])), t(sc["future"](A["future"][rows])), t(sc["static"](A["static"][rows])),
            t(((A["y"][rows] - A["level"][rows][:, None]) / A["spread"][rows][:, None] - sc["y_m"]) / sc["y_s"]))


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class _CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self._pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self._pad, dilation=dilation)

    def forward(self, x):
        y = self.conv(x)
        return y[:, :, :-self._pad] if self._pad else y


class _TCNResBlock(nn.Module):
    def __init__(self, ch, kernel_size, dilation, dropout):
        super().__init__()
        g = min(8, ch)
        self.net = nn.Sequential(
            _CausalConv1d(ch, ch, kernel_size, dilation), nn.GroupNorm(g, ch), nn.GELU(), nn.Dropout(dropout),
            _CausalConv1d(ch, ch, kernel_size, dilation), nn.GroupNorm(g, ch), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x) + x


class _SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class Encoder(nn.Module):
    """Past block (B, 168, Fp) -> context (B, hidden)."""

    def __init__(self, kind, f_past, hidden, layers, dropout):
        super().__init__()
        self.kind = kind
        if kind == "lstm":
            self.norm = nn.LayerNorm(f_past)
            self.net = nn.LSTM(f_past, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        elif kind == "tcn":
            # dilations 1..32, kernel 3: receptive field 253 h > 168 h
            self.proj = nn.Conv1d(f_past, hidden, 1)
            self.net = nn.Sequential(*[_TCNResBlock(hidden, 3, d, dropout) for d in [1, 2, 4, 8, 16, 32]])
        elif kind == "transformer":
            self.proj = nn.Linear(f_past, hidden)
            self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
            nn.init.trunc_normal_(self.cls, std=0.02)
            self.pos = _SinusoidalPE(hidden, PAST_DAYS * 24 + 1, dropout)
            layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=2 * hidden, dropout=dropout,
                                               batch_first=True, norm_first=True, activation="gelu")
            self.net = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        else:
            raise ValueError(kind)

    def forward(self, x):
        if self.kind == "lstm":
            out, _ = self.net(self.norm(x))
            return out[:, -1]
        if self.kind == "tcn":
            return self.net(self.proj(x.transpose(1, 2)))[:, :, -1]
        e = self.proj(x)
        e = self.pos(torch.cat([self.cls.expand(x.size(0), -1, -1), e], dim=1))
        return self.net(e)[:, 0]


class DayAheadNet(nn.Module):
    def __init__(self, kind, f_past, f_future, n_static, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.encoder = Encoder(kind, f_past, hidden, layers, dropout)
        n_side = 24 * f_future + n_static
        self.head = nn.Sequential(nn.LayerNorm(hidden + n_side), nn.Linear(hidden + n_side, 2 * hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(2 * hidden, 24))
        self.shortcut = nn.Linear(24 + n_side, 24)

    def forward(self, past, future, static):
        side = torch.cat([future.flatten(1), static], dim=1)
        h = self.encoder(past)
        return self.head(torch.cat([h, side], dim=1)) + self.shortcut(torch.cat([past[:, -24:, 0], side], dim=1))


# ---------------------------------------------------------------------------
# Training: AdamW, warm-up + cosine, L1 loss, early stopping
# ---------------------------------------------------------------------------

def fit(kind, train, val, seed, max_epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    past, future, static, y = [t.to(DEVICE) for t in train]
    if val is not None:
        vpast, vfut, vsta, vy = [t.to(DEVICE) for t in val]
    net = DayAheadNet(kind, past.shape[2], future.shape[2], static.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR[kind], weight_decay=1e-4)
    steps = math.ceil(len(y) / BATCH)
    total, warm = max_epochs * steps, WARMUP_EPOCHS * steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else max(1e-6, 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))))
    loss_fn = nn.L1Loss()
    best, best_state, wait, best_epoch = np.inf, None, 0, max_epochs
    for epoch in range(1, max_epochs + 1):
        net.train()
        perm = torch.randperm(len(y), device=DEVICE)
        for k in range(0, len(y), BATCH):
            b = perm[k:k + BATCH]
            opt.zero_grad()
            loss = loss_fn(net(past[b], future[b], static[b]), y[b])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        net.eval()
        if val is None:                     # fixed number of epochs (refit after early stopping), no validation
            continue
        with torch.no_grad():
            v = loss_fn(net(vpast, vfut, vsta), vy).item()
        if v < best - 1e-5:
            best, wait, best_epoch = v, 0, epoch
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if val is not None:
        net.load_state_dict(best_state)
    net.best_epoch = best_epoch            # epoch with the lowest validation loss (max_epochs without validation)
    net.eval()
    return net, epoch, best


def predict(net, data, sc, level, spread):
    past, future, static, _ = [t.to(DEVICE) for t in data]
    with torch.no_grad():
        z = net(past, future, static).cpu().numpy()
    return (z * sc["y_s"] + sc["y_m"]) * spread[:, None] + level[:, None]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def long_format(A, rows, P, split, model):
    d = A["dates"][rows]
    return pd.DataFrame({"date": np.repeat(d, 24), "hour": np.tile(np.arange(24), len(d)), "split": split, "model": model,
                         "y_true": A["y"][rows].ravel(), "y_pred": P.ravel(), "scored": A["mask"][rows].ravel()})

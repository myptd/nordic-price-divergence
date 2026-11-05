"""
Bidding zones used in the study.

All collectors and the preprocessing script read this file, so zones are
defined in one place. Norway is the focus; Sweden and Germany are controls.
Denmark (DK1, DK2) and the Netherlands (NL) are price-only zones (role
"price_only"): only their day-ahead prices are downloaded and processed, with
no load, wind/solar, weather or reservoir data. They are not modelling zones.

For each zone:
  price_areas  ENTSO-E areas for day-ahead prices as (area, start, end).
               Germany changed bidding zone on 2018-10-01 (DE-AT-LU -> DE-LU),
               so its price series is spliced from two areas.
  load_areas   ENTSO-E areas for load and wind/solar forecasts, in priority
               order. Later areas only fill gaps in earlier ones.
  weather      (name, latitude, longitude) points, averaged per zone.
  reservoir_nr    NVE Elspot area number (Norway only).
  reservoir_area  ENTSO-E area for the weekly reservoir filling (data item
                  16.1.D). Norway and Sweden; for Norway it is a cross-check
                  of the NVE data. Germany has no meaningful reservoir hydro.
"""

ZONES = {
    # Norway
    "NO1": {"country": "NO", "role": "focus",
            "price_areas": [("NO_1", None, None)], "load_areas": ["NO_1"],
            "weather": [("oslo", 59.91, 10.75)], "reservoir_nr": 1, "reservoir_area": "NO_1"},
    "NO2": {"country": "NO", "role": "focus",
            "price_areas": [("NO_2", None, None)], "load_areas": ["NO_2"],
            "weather": [("kristiansand", 58.15, 7.99)], "reservoir_nr": 2, "reservoir_area": "NO_2"},
    "NO3": {"country": "NO", "role": "focus",
            "price_areas": [("NO_3", None, None)], "load_areas": ["NO_3"],
            "weather": [("trondheim", 63.43, 10.39)], "reservoir_nr": 3, "reservoir_area": "NO_3"},
    "NO4": {"country": "NO", "role": "focus",
            "price_areas": [("NO_4", None, None)], "load_areas": ["NO_4"],
            "weather": [("tromso", 69.65, 18.96)], "reservoir_nr": 4, "reservoir_area": "NO_4"},
    "NO5": {"country": "NO", "role": "focus",
            "price_areas": [("NO_5", None, None)], "load_areas": ["NO_5"],
            "weather": [("bergen", 60.39, 5.32)], "reservoir_nr": 5, "reservoir_area": "NO_5"},

    # Sweden (control). Reservoir filling from ENTSO-E only (no NVE equivalent).
    "SE1": {"country": "SE", "role": "control",
            "price_areas": [("SE_1", None, None)], "load_areas": ["SE_1"],
            "weather": [("lulea", 65.58, 22.15)], "reservoir_nr": None, "reservoir_area": "SE_1"},
    "SE2": {"country": "SE", "role": "control",
            "price_areas": [("SE_2", None, None)], "load_areas": ["SE_2"],
            "weather": [("sundsvall", 62.39, 17.31), ("ostersund", 63.18, 14.64)],
            "reservoir_nr": None, "reservoir_area": "SE_2"},
    "SE3": {"country": "SE", "role": "control",
            "price_areas": [("SE_3", None, None)], "load_areas": ["SE_3"],
            "weather": [("stockholm", 59.33, 18.07), ("gothenburg", 57.71, 11.97)],
            "reservoir_nr": None, "reservoir_area": "SE_3"},
    "SE4": {"country": "SE", "role": "control",
            "price_areas": [("SE_4", None, None)], "load_areas": ["SE_4"],
            "weather": [("malmo", 55.60, 13.00)], "reservoir_nr": None, "reservoir_area": "SE_4"},

    # Germany (control). Load/wind/solar from DE-LU, gaps filled from DE
    # (Luxembourg is <1% of load).
    "DE_LU": {"country": "DE", "role": "control",
              "price_areas": [("DE_AT_LU", None, "2018-10-01"), ("DE_LU", "2018-10-01", None)],
              "load_areas": ["DE_LU", "DE"],
              "weather": [("hamburg", 53.55, 9.99), ("bremerhaven", 53.54, 8.58),
                          ("berlin", 52.52, 13.40), ("cologne", 50.94, 6.96),
                          ("frankfurt", 50.11, 8.68), ("munich", 48.14, 11.58)],
              "reservoir_nr": None, "reservoir_area": None},
    # Price-only zones: day-ahead prices only. All three share CET/CEST with the zones above.
    "DK1": {"country": "DK", "role": "price_only",
            "price_areas": [("DK_1", None, None)], "load_areas": [], "weather": [],
            "reservoir_nr": None, "reservoir_area": None},
    "DK2": {"country": "DK", "role": "price_only",
            "price_areas": [("DK_2", None, None)], "load_areas": [], "weather": [],
            "reservoir_nr": None, "reservoir_area": None},
    "NL": {"country": "NL", "role": "price_only",
           "price_areas": [("NL", None, None)], "load_areas": [], "weather": [],
           "reservoir_nr": None, "reservoir_area": None},
}

NO_ZONES = [z for z in ZONES if ZONES[z]["country"] == "NO"]
PRICE_ONLY_ZONES = [z for z in ZONES if ZONES[z]["role"] == "price_only"]

# Raw downloads start one month early so price lags exist from 2018-01-01.
START = "2017-12-01"
END = "2026-01-01"  # exclusive

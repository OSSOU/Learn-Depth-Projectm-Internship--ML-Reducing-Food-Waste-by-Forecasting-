from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .config import DATE_COL, TARGET, GROUP_COLS

COMMON_MAP = {
    "week": "week",
    "center_id": "location_id",
    "meal_id": "item_id",
    "num_orders": "demand",
    "emailer_for_promotion": "promotion",
    "homepage_featured": "featured",
    "checkout_price": "price",
    "base_price": "base_price",
}

def make_synthetic(days=730, locations=3, items=6, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    rows = []
    holiday_days = set(pd.to_datetime([
        "2024-01-01", "2024-01-26", "2024-03-25", "2024-04-14", "2024-05-01",
        "2024-08-15", "2024-10-02", "2024-10-31", "2024-11-01", "2024-12-25",
        "2025-01-01", "2025-01-26", "2025-03-14", "2025-04-14", "2025-05-01",
        "2025-08-15", "2025-10-02", "2025-10-20", "2025-12-25",
    ]).date)
    event_dates = set(pd.date_range("2024-02-10", "2025-11-30", freq="45D").date)
    for loc in range(1, locations + 1):
        for item in range(1, items + 1):
            item_base = 55 + 13 * item + rng.normal(0, 4)
            loc_factor = 0.75 + 0.25 * loc
            for d in dates:
                dow = d.dayofweek
                weekend = int(dow >= 5)
                holiday = int(d.date() in holiday_days)
                event = int(d.date() in event_dates and loc in {1, 2})
                promotion = int(rng.random() < (0.10 + 0.18 * event))
                academic_break = int((d.month in [5, 6]) or (d.month == 12 and d.day >= 15))
                temp = 25 + 7*np.sin(2*np.pi*(d.dayofyear-80)/365) + rng.normal(0, 2.2)
                rain = max(0, rng.gamma(1.4, 3.0) * (1.8 if d.month in [6,7,8,9] else 0.5))
                event_intensity = float(event * rng.uniform(0.5, 1.5))
                weather_stress = float(abs(temp - 25)/10 + rain/30)
                payday = int(d.day in [1, 2, 3, 15, 16])
                season = 1 + 0.10*np.sin(2*np.pi*d.dayofyear/365) + 0.05*np.cos(2*np.pi*d.dayofyear/365)
                weekly = [1.00, 0.98, 1.02, 1.04, 1.15, 1.30, 1.18][dow]
                context = 1 + 0.14*promotion + 0.22*event_intensity + 0.08*payday - 0.08*holiday - 0.05*academic_break
                weather = 1 - 0.06*min(weather_stress, 1.5)
                demand_mean = item_base * loc_factor * season * weekly * context * weather
                noise = rng.normal(0, max(4, demand_mean*0.09))
                demand = max(0, demand_mean + noise)
                rows.append({
                    "date": d, "location_id": f"L{loc}", "item_id": f"I{item}",
                    "demand": round(demand, 2), "is_holiday": holiday,
                    "is_event": event, "promotion": promotion,
                    "temperature": round(temp, 2), "rain_mm": round(rain, 2),
                    "academic_break": academic_break, "payday_week": payday,
                    "event_intensity": round(event_intensity, 3),
                    "weather_stress": round(weather_stress, 3),
                })
    return pd.DataFrame(rows)

def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {k: v for k, v in COMMON_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    if "date" not in df.columns and "week" in df.columns:
        min_week = int(df["week"].min())
        df["date"] = pd.Timestamp("2016-01-01") + pd.to_timedelta((df["week"].astype(int)-min_week)*7, unit="D")
    required = {"date", "location_id", "item_id", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    for c, default in {
        "is_holiday":0, "is_event":0, "promotion":0, "temperature":25,
        "rain_mm":0, "academic_break":0, "payday_week":0, "event_intensity":0,
        "weather_stress":0,
    }.items():
        if c not in df.columns:
            df[c] = default
    return df.sort_values(GROUP_COLS + [DATE_COL]).reset_index(drop=True)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(GROUP_COLS + [DATE_COL])
    dt = out[DATE_COL]
    out["day_of_week"] = dt.dt.dayofweek
    out["day_of_month"] = dt.dt.day
    out["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    out["month"] = dt.dt.month
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    out["is_month_start"] = dt.dt.is_month_start.astype(int)
    out["is_month_end"] = dt.dt.is_month_end.astype(int)
    g = out.groupby(GROUP_COLS, group_keys=False)[TARGET]
    for lag in [1, 7, 14, 28]:
        out[f"lag_{lag}"] = g.shift(lag)
    out["rolling_mean_7"] = out.groupby(GROUP_COLS)[TARGET].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    out["rolling_mean_28"] = out.groupby(GROUP_COLS)[TARGET].transform(lambda s: s.shift(1).rolling(28, min_periods=7).mean())
    out["rolling_std_28"] = out.groupby(GROUP_COLS)[TARGET].transform(lambda s: s.shift(1).rolling(28, min_periods=7).std())
    out["trend_7_28"] = out["rolling_mean_7"] / out["rolling_mean_28"].replace(0, np.nan)
    return out.dropna(subset=["lag_28", "rolling_mean_28"]).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--locations", type=int, default=3)
    ap.add_argument("--items", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="data/raw/demand.csv")
    args = ap.parse_args()
    df = make_synthetic(args.days, args.locations, args.items, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df):,} rows to {args.output}")

if __name__ == "__main__":
    main()

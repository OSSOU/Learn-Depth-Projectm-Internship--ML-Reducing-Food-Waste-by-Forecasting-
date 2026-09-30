from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

for p in [DATA_RAW, DATA_PROCESSED, MODELS, REPORTS]:
    p.mkdir(parents=True, exist_ok=True)

TARGET = "demand"
DATE_COL = "date"
GROUP_COLS = ["location_id", "item_id"]
BASE_FEATURES = [
    "day_of_week", "day_of_month", "week_of_year", "month", "is_weekend",
    "is_month_start", "is_month_end", "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28", "rolling_std_28", "trend_7_28"
]
CONTEXT_FEATURES = [
    "is_holiday", "is_event", "promotion", "temperature", "rain_mm",
    "academic_break", "payday_week", "event_intensity", "weather_stress"
]
FEATURES = BASE_FEATURES + CONTEXT_FEATURES

# ML-T2-003 — Reducing Food Waste by Forecasting Preparation Demand Under Changing Conditions

A zero-cost, research-oriented forecasting system aligned with the Learn Depth ML-T2-003 challenge.

## Research framing
The challenge asks whether demand can be predicted accurately enough to reduce avoidable preparation without increasing shortage risk, while studying contextual variables, unusual days, time-aware evaluation, and decision consequences. The supplied challenge document specifies no particular dataset, algorithm, feature set, architecture, or metric; this implementation therefore makes those choices explicit and reproducible. fileciteturn0file0L149-L213

## What this repository contains
- A defensible synthetic food-service demand generator for reproducible experiments.
- Optional support for the common Food Demand Forecasting CSV schema (`week`, `center_id`, `meal_id`, `checkout_price`, `base_price`, `emailer_for_promotion`, `homepage_featured`, `num_orders`, etc.).
- Time-aware train/validation/test splits; no random temporal split.
- Baselines: seasonal naive and moving-average.
- Contextual ML model: HistGradientBoostingRegressor.
- Quantile models for prediction intervals.
- Feature ablation experiment to test whether contextual variables add value.
- Backtesting by chronological folds.
- Ordinary vs unusual-day analysis.
- Asymmetric preparation decision simulation with separate shortage and waste costs.
- Streamlit dashboard for operational use.
- CLI scripts for generation, training, evaluation, backtesting, and forecasting.

## Installation
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Fast start
```bash
python -m src.generate_data --days 730 --locations 3 --items 6
python -m src.train
python -m src.evaluate
python -m src.backtest
streamlit run app/streamlit_app.py
```

Open the displayed Streamlit URL in a browser.

## Use a real CSV
Place a CSV at `data/raw/demand.csv` and run:
```bash
python -m src.train --input data/raw/demand.csv
```

The loader recognizes either:
1. The project's canonical schema: `date`, `location_id`, `item_id`, `demand` plus contextual columns; or
2. The common Food Demand Forecasting schema. In the latter case, `week` is converted into a date index and `num_orders` becomes demand. Because that dataset is weekly and lacks weather, holiday, and event variables, the project will only use the contextual columns actually present rather than pretending absent information exists.

## Research experiments
### Experiment A — baseline vs contextual model
Compare seasonal naive/moving average against the ML model using chronological test data.

### Experiment B — contextual ablation
Run:
```bash
python -m src.ablation
```
This compares a calendar/history-only feature set against a contextual feature set. The result directly addresses the research question about whether context genuinely improves forecasts.

### Experiment C — unusual periods
The synthetic generator marks unusual days such as holidays, events, promotions, and heat/rain conditions. `evaluate.py` reports ordinary and unusual performance separately.

### Experiment D — operational decision
The forecast is converted into a preparation quantity using asymmetric shortage/waste costs. Under the default policy, the target service level is derived from the relative shortage and waste costs rather than treating MAE as the only objective.

## Important limitation
The synthetic data are useful for engineering and controlled experimentation, not evidence about real food-service demand. For a research submission, replace or supplement them with a legally usable real dataset and document its provenance, licensing, temporal coverage, and missing variables.

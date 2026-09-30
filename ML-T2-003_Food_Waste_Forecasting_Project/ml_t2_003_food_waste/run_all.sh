#!/usr/bin/env bash
set -e
python -m src.generate_data --days 730 --locations 3 --items 6
python -m src.train
python -m src.evaluate
python -m src.backtest
python -m src.ablation
python -m src.plot_results

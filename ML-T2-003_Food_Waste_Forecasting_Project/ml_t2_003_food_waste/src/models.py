from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

class SeasonalNaive:
    def __init__(self, lag=7): self.lag = lag
    def fit(self, X, y):
        self.last = float(np.asarray(y)[-self.lag:].mean()); return self
    def predict(self, X): return np.full(len(X), self.last)

class MovingAverage:
    def __init__(self, window=7): self.window=window
    def fit(self, X, y): self.value=float(np.asarray(y)[-self.window:].mean()); return self
    def predict(self, X): return np.full(len(X), self.value)

def make_model(loss="squared_error", quantile=None, random_state=42):
    if quantile is not None:
        return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, max_iter=300, learning_rate=0.05,
            max_leaf_nodes=31, l2_regularization=1.0, random_state=random_state)
    return HistGradientBoostingRegressor(loss=loss, max_iter=350, learning_rate=0.05,
        max_leaf_nodes=31, l2_regularization=1.0, random_state=random_state)

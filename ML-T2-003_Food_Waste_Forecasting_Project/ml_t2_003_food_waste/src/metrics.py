import numpy as np

def mae(y, p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
def mape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(np.mean(np.abs((y-p)/np.maximum(np.abs(y), 1e-6))) * 100)
def wape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(np.sum(np.abs(y-p))/np.maximum(np.sum(np.abs(y)), 1e-6) * 100)
def pinball(y, q, alpha):
    y, q = np.asarray(y), np.asarray(q)
    e = y-q
    return float(np.mean(np.maximum(alpha*e, (alpha-1)*e)))
def decision_cost(y, prep, shortage_cost=3.0, waste_cost=1.0):
    y, prep = np.asarray(y), np.asarray(prep)
    shortage = np.maximum(y-prep, 0)
    waste = np.maximum(prep-y, 0)
    return float(np.mean(shortage_cost*shortage + waste_cost*waste))
def service_level(y, prep):
    y, prep = np.asarray(y), np.asarray(prep)
    return float(np.mean(y <= prep))

def summary(y, p):
    return {"MAE":mae(y,p),"RMSE":rmse(y,p),"WAPE_pct":wape(y,p)}

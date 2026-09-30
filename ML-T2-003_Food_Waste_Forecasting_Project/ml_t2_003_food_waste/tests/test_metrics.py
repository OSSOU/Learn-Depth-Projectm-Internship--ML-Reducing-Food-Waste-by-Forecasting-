import numpy as np
from src.metrics import mae, rmse, decision_cost, service_level

def test_basic_metrics():
    y=np.array([10,20,30]); p=np.array([10,25,25])
    assert mae(y,p)==10/3
    assert round(rmse(y,p),6)==round(np.sqrt(50/3),6)

def test_decision_cost_and_service():
    y=np.array([10,20]); p=np.array([12,18])
    assert decision_cost(y,p,3,1)==4.0
    assert service_level(y,p)==0.5

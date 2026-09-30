from __future__ import annotations
import json
from pathlib import Path
import joblib, pandas as pd
from .metrics import summary, decision_cost, service_level, pinball
from .config import REPORTS, BASE_FEATURES, CONTEXT_FEATURES

def main():
    p=Path("data/processed/test_predictions.csv")
    if not p.exists():
        from .train import main as train_main; train_main()
    df=pd.read_csv(p)
    meta=json.loads(Path("models/metadata.json").read_text())
    pred=df.pred.values; y=df.demand.values
    rows=[]
    for name, mask in [("all", pd.Series(True,index=df.index)), ("unusual", (df.is_holiday==1)|(df.is_event==1)|(df.promotion==1)|(df.weather_stress>=1.0)), ("ordinary", ~((df.is_holiday==1)|(df.is_event==1)|(df.promotion==1)|(df.weather_stress>=1.0)))]:
        yy=df.loc[mask,"demand"]; pp=df.loc[mask,"pred"]
        s=summary(yy,pp); s.update(period=name,n=len(yy),decision_cost=decision_cost(yy,pp),service_level=service_level(yy,pp))
        rows.append(s)
    # Quantile interval and asymmetric decision policy.
    q10,q90=df.q10.values,df.q90.values
    target_alpha=3/(3+1)  # shortage / (shortage + waste)
    prep=q90 if target_alpha>=0.9 else pred
    rows.append({"period":"decision_policy","n":len(y),"MAE":summary(y,prep)["MAE"],"RMSE":summary(y,prep)["RMSE"],"WAPE_pct":summary(y,prep)["WAPE_pct"],"decision_cost":decision_cost(y,prep),"service_level":service_level(y,prep),"target_service_level":target_alpha})
    out=pd.DataFrame(rows); REPORTS.mkdir(exist_ok=True); out.to_csv(REPORTS/"evaluation.csv",index=False)
    print(out.to_string(index=False))
    print("90% interval coverage:", ((y>=q10)&(y<=q90)).mean())
    print("Q10 pinball:", pinball(y,q10,0.10), "Q90 pinball:", pinball(y,q90,0.90))

if __name__=="__main__": main()

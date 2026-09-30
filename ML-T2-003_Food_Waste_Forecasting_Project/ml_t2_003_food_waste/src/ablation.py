from __future__ import annotations
import pandas as pd
from .data import load_data, add_features, make_synthetic
from .config import BASE_FEATURES, CONTEXT_FEATURES
from .models import make_model
from .metrics import summary, decision_cost

def main():
    try: df=load_data("data/raw/demand.csv")
    except FileNotFoundError: df=make_synthetic()
    df=add_features(df)
    for c in ["location_id","item_id"]: df[c+"_code"]=df[c].astype("category").cat.codes
    features0=BASE_FEATURES+["location_id_code","item_id_code"]
    features1=features0+CONTEXT_FEATURES
    cut=df.date.max()-pd.Timedelta(days=60); train=df[df.date<cut]; test=df[df.date>=cut]
    rows=[]
    for name,features in [("history_calendar_only",features0),("history_plus_context",features1)]:
        m=make_model(); m.fit(train[features],train.demand); p=m.predict(test[features]); s=summary(test.demand,p); s.update(model=name,decision_cost=decision_cost(test.demand,p)); rows.append(s)
    out=pd.DataFrame(rows); out.to_csv("reports/ablation.csv",index=False); print(out.to_string(index=False))

if __name__=="__main__": main()

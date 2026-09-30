from __future__ import annotations
import pandas as pd, numpy as np
from .data import load_data, add_features, make_synthetic
from .config import FEATURES, REPORTS
from .models import make_model
from .metrics import summary, decision_cost

def main():
    try: df=load_data("data/raw/demand.csv")
    except FileNotFoundError: df=make_synthetic()
    df=add_features(df)
    for c in ["location_id","item_id"]: df[c+"_code"]=df[c].astype("category").cat.codes
    features=FEATURES+["location_id_code","item_id_code"]
    dates=sorted(df.date.dt.date.unique())
    folds=[]
    for i in range(180,len(dates),120):
        train_end=pd.Timestamp(dates[i-1]); test_end=pd.Timestamp(dates[min(i+29,len(dates)-1)])
        train=df[df.date<=train_end]; test=df[(df.date>train_end)&(df.date<=test_end)]
        if len(test)==0: continue
        model=make_model(); model.fit(train[features],train.demand); pred=model.predict(test[features])
        s=summary(test.demand,pred); s.update(fold=len(folds)+1,train_end=str(train_end),test_end=str(test_end),decision_cost=decision_cost(test.demand,pred))
        folds.append(s)
    out=pd.DataFrame(folds); out.to_csv(REPORTS/"backtest.csv",index=False); print(out.to_string(index=False))

if __name__=="__main__": main()

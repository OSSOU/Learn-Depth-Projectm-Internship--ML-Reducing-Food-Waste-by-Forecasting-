from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import pandas as pd
from .config import FEATURES, BASE_FEATURES, MODELS
from .data import load_data, add_features, make_synthetic
from .models import make_model

def temporal_split(df, test_days=60, val_days=60):
    max_date=df.date.max()
    test_start=max_date-pd.Timedelta(days=test_days-1)
    val_start=test_start-pd.Timedelta(days=val_days)
    train=df[df.date<val_start]
    val=df[(df.date>=val_start)&(df.date<test_start)]
    test=df[df.date>=test_start]
    return train,val,test

def train_one(train, features, quantile=None):
    x=train[features].copy(); y=train.demand
    model=make_model(quantile=quantile)
    model.fit(x,y)
    return model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/demand.csv")
    ap.add_argument("--test-days", type=int, default=60)
    ap.add_argument("--val-days", type=int, default=60)
    args=ap.parse_args()
    p=Path(args.input)
    if p.exists(): df=load_data(p)
    else: df=make_synthetic()
    df=add_features(df)
    # Time split is global because deployment forecasting is chronological.
    train,val,test=temporal_split(df,args.test_days,args.val_days)
    # HistGradientBoosting cannot consume strings; one-hot-free representation uses numeric stable IDs.
    all_df=pd.concat([train,val,test])
    for c in ["location_id","item_id"]:
        codes=all_df[c].astype("category").cat.codes
        all_df[c+"_code"]=codes
    extra=["location_id_code","item_id_code"]
    train=all_df.loc[train.index]; val=all_df.loc[val.index]; test=all_df.loc[test.index]
    features=FEATURES+extra
    model=train_one(train,features)
    low=train_one(train,features,0.10)
    high=train_one(train,features,0.90)
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model":model,"features":features,"model_type":"hgb_mean"},MODELS/"demand_model.joblib")
    joblib.dump({"model":low,"features":features,"model_type":"hgb_q10"},MODELS/"demand_q10.joblib")
    joblib.dump({"model":high,"features":features,"model_type":"hgb_q90"},MODELS/"demand_q90.joblib")
    meta={"train_end":str(train.date.max()),"val_start":str(val.date.min()),"val_end":str(val.date.max()),"test_start":str(test.date.min()),"test_end":str(test.date.max()),"features":features}
    (MODELS/"metadata.json").write_text(json.dumps(meta,indent=2))
    test.assign(pred=model.predict(test[features]), q10=low.predict(test[features]), q90=high.predict(test[features])).to_csv("data/processed/test_predictions.csv",index=False)
    print(json.dumps(meta,indent=2))

if __name__=="__main__": main()

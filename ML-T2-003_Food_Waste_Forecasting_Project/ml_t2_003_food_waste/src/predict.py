from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, pandas as pd
from .data import load_data, add_features
from .config import MODELS

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--output",default="data/processed/forecast.csv"); args=ap.parse_args()
    df=add_features(load_data(args.input))
    for c in ["location_id","item_id"]: df[c+"_code"]=df[c].astype("category").cat.codes
    model=joblib.load(MODELS/"demand_model.joblib"); low=joblib.load(MODELS/"demand_q10.joblib"); high=joblib.load(MODELS/"demand_q90.joblib")
    df["forecast"]=model["model"].predict(df[model["features"]]); df["q10"]=low["model"].predict(df[low["features"]]); df["q90"]=high["model"].predict(df[high["features"]])
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); df.to_csv(args.output,index=False); print(df.tail().to_string(index=False))

if __name__=="__main__": main()

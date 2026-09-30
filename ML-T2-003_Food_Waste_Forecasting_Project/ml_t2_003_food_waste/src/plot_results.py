import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    p=Path("data/processed/test_predictions.csv")
    df=pd.read_csv(p,parse_dates=["date"])
    agg=df.groupby("date")[["demand","pred","q10","q90"]].mean()
    plt.figure(figsize=(14,6)); plt.plot(agg.index,agg.demand,label="Actual"); plt.plot(agg.index,agg.pred,label="Forecast"); plt.fill_between(agg.index,agg.q10,agg.q90,alpha=.2,label="80% interval"); plt.legend(); plt.title("Test-period demand forecast"); plt.tight_layout(); plt.savefig("reports/test_forecast.png",dpi=160); plt.close()

if __name__=="__main__": main()

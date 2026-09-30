from __future__ import annotations
import argparse
from pathlib import Path
import joblib, numpy as np, pandas as pd
from .config import MODELS
from .data import load_data

def build_next_row(df, location_id, item_id, context):
    hist=df[(df.location_id==location_id)&(df.item_id==item_id)].sort_values('date').copy()
    if len(hist)<28: raise ValueError('At least 28 historical observations are required for lag features.')
    last=hist.iloc[-1]
    d=pd.Timestamp(last.date)+pd.Timedelta(days=1)
    demand=hist.demand
    row={
        'date':d,'location_id':location_id,'item_id':item_id,
        'day_of_week':d.dayofweek,'day_of_month':d.day,'week_of_year':int(d.isocalendar().week),
        'month':d.month,'is_weekend':int(d.dayofweek>=5),'is_month_start':int(d.is_month_start),'is_month_end':int(d.is_month_end),
        'lag_1':float(demand.iloc[-1]),'lag_7':float(demand.iloc[-7]),'lag_14':float(demand.iloc[-14]),'lag_28':float(demand.iloc[-28]),
        'rolling_mean_7':float(demand.iloc[-7:].mean()),'rolling_mean_28':float(demand.iloc[-28:].mean()),
        'rolling_std_28':float(demand.iloc[-28:].std()),'trend_7_28':float(demand.iloc[-7:].mean()/max(demand.iloc[-28:].mean(),1e-6)),
    }
    row.update(context); row['location_id_code']=int(pd.Categorical([location_id],categories=sorted(df.location_id.unique())).codes[0]); row['item_id_code']=int(pd.Categorical([item_id],categories=sorted(df.item_id.unique())).codes[0])
    return pd.DataFrame([row])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='data/raw/demand.csv'); ap.add_argument('--location',required=True); ap.add_argument('--item',required=True); ap.add_argument('--holiday',type=int,default=0); ap.add_argument('--event',type=int,default=0); ap.add_argument('--promotion',type=int,default=0); ap.add_argument('--temperature',type=float,default=25); ap.add_argument('--rain-mm',type=float,default=0); ap.add_argument('--academic-break',type=int,default=0); ap.add_argument('--payday-week',type=int,default=0); ap.add_argument('--event-intensity',type=float,default=0); ap.add_argument('--weather-stress',type=float,default=0); args=ap.parse_args()
    df=load_data(args.input)
    context={'is_holiday':args.holiday,'is_event':args.event,'promotion':args.promotion,'temperature':args.temperature,'rain_mm':args.rain_mm,'academic_break':args.academic_break,'payday_week':args.payday_week,'event_intensity':args.event_intensity,'weather_stress':args.weather_stress}
    row=build_next_row(df,args.location,args.item,context)
    m=joblib.load(MODELS/'demand_model.joblib'); lo=joblib.load(MODELS/'demand_q10.joblib'); hi=joblib.load(MODELS/'demand_q90.joblib')
    print(row[['date','location_id','item_id']].to_string(index=False)); print('Forecast:',float(m['model'].predict(row[m['features']])[0])); print('Q10:',float(lo['model'].predict(row[lo['features']])[0])); print('Q90:',float(hi['model'].predict(row[hi['features']])[0]))
if __name__=='__main__': main()

from __future__ import annotations
from pathlib import Path
import sys
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data import load_data
from src.forecast_next import build_next_row

st.set_page_config(page_title="Food Demand Forecasting",layout="wide")
st.title("ML-T2-003 — Food Preparation Demand Forecasting")
st.caption("Decision-oriented next-day forecasting under changing conditions.")

if not (ROOT/"models/demand_model.joblib").exists():
    st.warning("Train the model first: python -m src.generate_data && python -m src.train")
    st.stop()

df=load_data(ROOT/"data/raw/demand.csv")
model=joblib.load(ROOT/"models/demand_model.joblib")
q10=joblib.load(ROOT/"models/demand_q10.joblib")
q90=joblib.load(ROOT/"models/demand_q90.joblib")

locations=sorted(df.location_id.unique()); items=sorted(df.item_id.unique())
loc=st.sidebar.selectbox("Location",locations); item=st.sidebar.selectbox("Item",items)
st.sidebar.header("Tomorrow's context")
holiday=st.sidebar.checkbox("Holiday")
event=st.sidebar.checkbox("Event")
promotion=st.sidebar.checkbox("Promotion")
temperature=st.sidebar.number_input("Temperature (°C)", -10.0, 50.0, 25.0)
rain=st.sidebar.number_input("Rain (mm)",0.0,300.0,0.0)
academic=st.sidebar.checkbox("Academic break")
payday=st.sidebar.checkbox("Payday week")
event_intensity=st.sidebar.slider("Event intensity",0.0,2.0,0.0)
weather_stress=st.sidebar.slider("Weather stress",0.0,3.0,0.0)
shortage_cost=st.sidebar.number_input("Shortage cost / unit",0.1,100.0,3.0,0.1)
waste_cost=st.sidebar.number_input("Waste cost / unit",0.1,100.0,1.0,0.1)

context={'is_holiday':int(holiday),'is_event':int(event),'promotion':int(promotion),'temperature':temperature,'rain_mm':rain,'academic_break':int(academic),'payday_week':int(payday),'event_intensity':event_intensity,'weather_stress':weather_stress}
row=build_next_row(df,loc,item,context)
forecast=float(model['model'].predict(row[model['features']])[0]); lo=float(q10['model'].predict(row[q10['features']])[0]); hi=float(q90['model'].predict(row[q90['features']])[0])
service=shortage_cost/(shortage_cost+waste_cost)
prep=hi if service>=0.90 else forecast

st.subheader(f"Forecast for {row.date.iloc[0].date()}")
c1,c2,c3,c4=st.columns(4); c1.metric("Expected demand",f"{forecast:.0f}"); c2.metric("Q10",f"{lo:.0f}"); c3.metric("Q90",f"{hi:.0f}"); c4.metric("Suggested preparation",f"{prep:.0f}")
st.info(f"Relative shortage/waste costs imply a target service level of {service:.0%}. With only Q10, mean and Q90 available, the prototype uses the mean unless the target is at least 90%, when it uses Q90.")

hist=df[(df.location_id==loc)&(df.item_id==item)].tail(120)
fig=go.Figure(); fig.add_trace(go.Scatter(x=hist.date,y=hist.demand,name="Actual")); fig.add_trace(go.Scatter(x=hist.date,y=hist.demand.rolling(7).mean(),name="7-day mean")); fig.update_layout(height=420,title="Recent demand history"); st.plotly_chart(fig,use_container_width=True)

st.subheader("Forecast inputs")
st.dataframe(row[['date','day_of_week','lag_1','lag_7','lag_14','lag_28','rolling_mean_7','rolling_mean_28','is_holiday','is_event','promotion','temperature','rain_mm','academic_break','payday_week','event_intensity','weather_stress']].T.rename(columns={row.index[0]:'value'}),use_container_width=True)

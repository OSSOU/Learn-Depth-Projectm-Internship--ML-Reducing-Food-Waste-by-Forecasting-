"""Notebook-style executable research workflow.
Run sections manually in Jupyter by copying cells or execute as a script.
"""
from src.data import make_synthetic, add_features
from src.models import make_model
from src.metrics import summary, decision_cost
import pandas as pd

df=add_features(make_synthetic())
for c in ["location_id","item_id"]: df[c+"_code"]=df[c].astype("category").cat.codes
features=[c for c in df.columns if c not in {"date","location_id","item_id","demand"}]
cut=df.date.max()-pd.Timedelta(days=60)
train=df[df.date<cut]; test=df[df.date>=cut]
model=make_model(); model.fit(train[features],train.demand); pred=model.predict(test[features])
print(summary(test.demand,pred)); print("Decision cost:",decision_cost(test.demand,pred))

import pandas as pd
from sklearn.linear_model import ElasticNet
import joblib

df=pd.read_csv("subject1.csv")
x=df.drop("Semester",axis=1)
y=df["Semester"]
model=ElasticNet(alpha=0.01,l1_ratio=0.75)
model.fit(x,y)

#Saving The Model
joblib.dump(model,"subject1.pkl")

#Loading Saved Model for Testing
model=joblib.load("subject1.pkl")
model.predict([[80,87,96]])

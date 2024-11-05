# RUN USING
# uvicorn main:app --reload 

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    features: list

# Load your model
model = joblib.load('./model/model.pkl')

@app.post('/predict')
def predict(request: PredictionRequest):
    df = pd.DataFrame([request.features])
    prediction = model.predict(df)
    return {'prediction': prediction.tolist()}
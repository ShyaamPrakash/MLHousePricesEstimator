from pathlib import Path

import joblib

import pandas as pd

from fastapi import FastAPI
from fastapi import HTTPException

from schema import(PredictionInput)

MODEL_PATH=Path('model.pkl')

if not MODEL_PATH.exists():
    raise RuntimeError('model.pkl is missing')

model=joblib.load(MODEL_PATH)

app=FastAPI(title='House Price estimator',version='1.0.0',description='A tool used to find the price of a particular house based on different criteria')

@app.get('/')
def lobby():
    return{'message':'Hi there welcome to the House price estimator'}

@app.get('/health')
def health():
    return{'health':'program running smooth','model':model is not None}



@app.post('/predict')
def predict(request: PredictionInput):
    data_dict = request.model_dump()
    
    data_dict['1stFlrSF'] = data_dict.pop('FirstFlrSF')
    data_dict['2ndFlrSF'] = data_dict.pop('SecondFlrSF')
    
    feature_order = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                     'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageCars', 'GarageArea']
    
    inp = pd.DataFrame([data_dict])
    
    inp = inp[feature_order]
    
    res = model.predict(inp)[0]
    return {"resultant_price": float(res)}



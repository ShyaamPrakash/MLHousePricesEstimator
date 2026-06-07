import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df=pd.read_csv('train.csv')


features=['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
            'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageCars', 'GarageArea']

target='SalePrice'
df=df[features+[target]].dropna()

X=df[features]
y=df[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

models={
    "linear_regression":Pipeline([('scaler',StandardScaler()),('lr',LinearRegression())]),
    "random_forest":Pipeline([('scaler',StandardScaler()),('rf',RandomForestRegressor())]),
    "xgboost":Pipeline([('scaler',StandardScaler()),('xgb',XGBRegressor())])
}


result={}
for name,model in models.items():
    model.fit(X_train,y_train)
    test_pred=model.predict(X_test)
    test_rmse=np.sqrt(mean_squared_error(y_test,test_pred))
    result[name]=test_rmse


best_model_name=min(result,key=lambda x:result[x])
best_model=models[best_model_name]
bp=best_model.predict(X_train)


print("The best model is:",best_model_name,"\nIts score is:",best_model.score(X_train,y_train))
joblib.dump(best_model,'model.pkl')
print('Best model saved successfully')


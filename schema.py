from pydantic import BaseModel

class PredictionInput(BaseModel):
    LotArea: float
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    TotalBsmtSF: float
    FirstFlrSF: float
    SecondFlrSF: float
    GrLivArea: float
    GarageCars: int
    GarageArea: float
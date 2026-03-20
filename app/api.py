from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import load_model, predict_single

app = FastAPI(title="Churn Prediction API")

model = load_model()


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: CustomerInput):
    return predict_single(model, customer.model_dump())
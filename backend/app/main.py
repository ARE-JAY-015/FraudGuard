from pydantic import BaseModel
from fastapi import HTTPException
from app import transaction_service

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.transaction_service import verify_transaction

class TransactionRequest(BaseModel):
    amount: float
    distance_from_home: float
    device_changed: int
    international: int
    failed_attempts: int
    merchant_category: str
    payment_method: str
    account_age_days: int
    customer_id: str

    
    
app = FastAPI(
    title="FraudGuard V3 API",
    description="AI-powered transaction fraud verification system",
    version="3.0.0"
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "success": True,
        "message": "FraudGuard V3 API is running",
        "version": "3.0.0"
    }


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "success": True,
        "status": "healthy"
    }


# ---------------------------------------------------------
# VERIFY TRANSACTION
# ---------------------------------------------------------

@app.get("/verify/{transaction_id}")
def verify(transaction_id: int):

    result = verify_transaction(transaction_id)

    if not result.get("success", False):
        raise HTTPException(
            status_code=404,
            detail=result.get(
                "message",
                "Transaction not found"
            )
        )

    return result
@app.post("/predict")
def predict_transaction(transaction: TransactionRequest):
    try:
        result = transaction_service.predict_new_transaction(
            transaction.dict()
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
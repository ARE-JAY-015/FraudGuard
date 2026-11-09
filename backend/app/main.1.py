from fastapi import FastAPI

app = FastAPI(
    title="FraudGuard API",
    description="ML-powered Fraud Detection System",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "FraudGuard API is running!",
        "status": "success"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

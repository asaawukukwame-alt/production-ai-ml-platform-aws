from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.predictor import predict_hos_risk
from app.services.storage import get_recent_predictions, save_prediction


router = APIRouter()


class HOSPredictionRequest(BaseModel):
    driving_hours_today: float = Field(..., ge=0, le=24)
    duty_window_hours: float = Field(..., ge=0, le=24)
    driving_hours_since_break: float = Field(..., ge=0, le=24)
    cycle_hours: float = Field(..., ge=0, le=100)
    cycle_limit: float = Field(70.0, ge=60, le=70)
    consecutive_off_duty_hours: float = Field(0.0, ge=0, le=168)


@router.get("/")
def root():
    return {
        "app": "TruckGuard AI",
        "message": "Federal trucking HOS risk and compliance engine",
        "status": "running",
        "docs": "/docs",
    }


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "truckguard-ai",
        "model": "Custom Naive Bayes HOS Risk Classifier",
        "database": "sqlite",
    }


@router.post("/predict")
def predict_hos(payload: HOSPredictionRequest):
    try:
        result = predict_hos_risk(payload.model_dump())
        saved_record = save_prediction(result)

        result["database_log"] = saved_record

        return result

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {error}",
        ) from error


@router.get("/predictions")
def list_predictions(limit: int = 10):
    try:
        return {
            "count": limit,
            "predictions": get_recent_predictions(limit=limit),
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not load predictions: {error}",
        ) from error

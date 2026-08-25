from pathlib import Path

import joblib
import pandas as pd

from app.models.train_model import FEATURE_COLUMNS, predict_one
from app.services.explanation import build_hos_explanation
from app.services.hos_rules import HOSInput, evaluate_hos


DEFAULT_MODEL_PATH = Path("models/hos_risk_model.pkl")

_MODEL_CACHE = None


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    global _MODEL_CACHE

    if _MODEL_CACHE is None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Run: python -m app.models.train_model"
            )

        _MODEL_CACHE = joblib.load(model_path)

    return _MODEL_CACHE


def build_feature_row(hos_input: HOSInput) -> pd.Series:
    return pd.Series(
        {
            "driving_hours_today": hos_input.driving_hours_today,
            "duty_window_hours": hos_input.duty_window_hours,
            "driving_hours_since_break": hos_input.driving_hours_since_break,
            "cycle_hours": hos_input.cycle_hours,
            "cycle_limit": hos_input.cycle_limit,
            "consecutive_off_duty_hours": hos_input.consecutive_off_duty_hours,
        }
    )


def predict_hos_risk(input_data: dict) -> dict:
    hos_input = HOSInput(
        driving_hours_today=float(input_data["driving_hours_today"]),
        duty_window_hours=float(input_data["duty_window_hours"]),
        driving_hours_since_break=float(
            input_data["driving_hours_since_break"]
        ),
        cycle_hours=float(input_data["cycle_hours"]),
        cycle_limit=float(input_data.get("cycle_limit", 70.0)),
        consecutive_off_duty_hours=float(
            input_data.get("consecutive_off_duty_hours", 0.0)
        ),
    )

    rules_result = evaluate_hos(hos_input)

    model = load_model()
    feature_row = build_feature_row(hos_input)

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in feature_row.index
    ]

    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    ml_prediction, ml_confidence = predict_one(model, feature_row)

    if rules_result["blocking_reasons"]:
        final_risk_level = "HIGH"
    else:
        final_risk_level = ml_prediction

    explanation = build_hos_explanation(
        rules_result=rules_result,
        ml_prediction=ml_prediction,
        ml_confidence=ml_confidence,
    )

    return {
        "input_data": {
            "driving_hours_today": hos_input.driving_hours_today,
            "duty_window_hours": hos_input.duty_window_hours,
            "driving_hours_since_break": hos_input.driving_hours_since_break,
            "cycle_hours": hos_input.cycle_hours,
            "cycle_limit": hos_input.cycle_limit,
            "consecutive_off_duty_hours": hos_input.consecutive_off_duty_hours,
        },
        "rules_result": rules_result,
        "ml_prediction": ml_prediction,
        "ml_confidence": round(float(ml_confidence), 4),
        "final_risk_level": final_risk_level,
        "explanation": explanation,
    }

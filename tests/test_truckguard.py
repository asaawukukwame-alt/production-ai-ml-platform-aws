from fastapi.testclient import TestClient

from app.services.hos_rules import HOSInput, evaluate_hos
from app.services.predictor import predict_hos_risk
from main import app


client = TestClient(app)


def test_low_risk_driver_can_continue():
    hos_input = HOSInput(
        driving_hours_today=5.5,
        duty_window_hours=7.0,
        driving_hours_since_break=4.0,
        cycle_hours=42.0,
        cycle_limit=70.0,
        consecutive_off_duty_hours=10.0,
    )

    result = evaluate_hos(hos_input)

    assert result["risk_level"] == "LOW"
    assert result["can_continue_driving"] is True
    assert result["break_required"] is False
    assert result["blocking_reasons"] == []


def test_medium_risk_driver_gets_warnings():
    hos_input = HOSInput(
        driving_hours_today=10.25,
        duty_window_hours=12.5,
        driving_hours_since_break=7.25,
        cycle_hours=67.0,
        cycle_limit=70.0,
        consecutive_off_duty_hours=10.0,
    )

    result = evaluate_hos(hos_input)

    assert result["risk_level"] == "MEDIUM"
    assert result["can_continue_driving"] is True
    assert len(result["warnings"]) >= 1


def test_high_risk_driver_cannot_continue():
    hos_input = HOSInput(
        driving_hours_today=11.0,
        duty_window_hours=13.0,
        driving_hours_since_break=8.0,
        cycle_hours=69.0,
        cycle_limit=70.0,
        consecutive_off_duty_hours=8.0,
    )

    result = evaluate_hos(hos_input)

    assert result["risk_level"] == "HIGH"
    assert result["can_continue_driving"] is False
    assert result["break_required"] is True
    assert len(result["blocking_reasons"]) >= 1


def test_predictor_returns_ml_and_rules_result():
    payload = {
        "driving_hours_today": 10.25,
        "duty_window_hours": 12.5,
        "driving_hours_since_break": 7.25,
        "cycle_hours": 67.0,
        "cycle_limit": 70.0,
        "consecutive_off_duty_hours": 10.0,
    }

    result = predict_hos_risk(payload)

    assert "rules_result" in result
    assert "ml_prediction" in result
    assert "ml_confidence" in result
    assert "final_risk_level" in result
    assert "explanation" in result
    assert result["final_risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "truckguard-ai"


def test_predict_endpoint():
    payload = {
        "driving_hours_today": 10.25,
        "duty_window_hours": 12.5,
        "driving_hours_since_break": 7.25,
        "cycle_hours": 67.0,
        "cycle_limit": 70.0,
        "consecutive_off_duty_hours": 10.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["final_risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "database_log" in data
    assert "prediction_id" in data["database_log"]


def test_predictions_endpoint():
    response = client.get("/predictions?limit=5")

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert isinstance(data["predictions"], list)

import json
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_URL = "sqlite:///truckguard_predictions.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    driving_hours_today = Column(Float, nullable=False)
    duty_window_hours = Column(Float, nullable=False)
    driving_hours_since_break = Column(Float, nullable=False)
    cycle_hours = Column(Float, nullable=False)
    cycle_limit = Column(Float, nullable=False)
    consecutive_off_duty_hours = Column(Float, nullable=False)

    rules_risk_level = Column(String(20), nullable=False)
    ml_prediction = Column(String(20), nullable=False)
    ml_confidence = Column(Float, nullable=False)
    final_risk_level = Column(String(20), nullable=False)
    can_continue_driving = Column(Boolean, nullable=False)

    summary = Column(Text, nullable=False)
    recommended_action = Column(Text, nullable=False)
    rules_result_json = Column(Text, nullable=False)
    full_result_json = Column(Text, nullable=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def save_prediction(result: dict) -> dict:
    init_db()

    input_data = result["input_data"]
    rules_result = result["rules_result"]
    explanation = result["explanation"]

    db = SessionLocal()

    try:
        log = PredictionLog(
            driving_hours_today=input_data["driving_hours_today"],
            duty_window_hours=input_data["duty_window_hours"],
            driving_hours_since_break=input_data["driving_hours_since_break"],
            cycle_hours=input_data["cycle_hours"],
            cycle_limit=input_data["cycle_limit"],
            consecutive_off_duty_hours=input_data["consecutive_off_duty_hours"],
            rules_risk_level=rules_result["risk_level"],
            ml_prediction=result["ml_prediction"],
            ml_confidence=result["ml_confidence"],
            final_risk_level=result["final_risk_level"],
            can_continue_driving=rules_result["can_continue_driving"],
            summary=explanation["summary"],
            recommended_action=explanation["recommended_action"],
            rules_result_json=json.dumps(rules_result),
            full_result_json=json.dumps(result),
        )

        db.add(log)
        db.commit()
        db.refresh(log)

        return {
            "prediction_id": log.id,
            "created_at": log.created_at.isoformat(),
        }

    finally:
        db.close()


def get_recent_predictions(limit: int = 10) -> list[dict]:
    init_db()

    db = SessionLocal()

    try:
        logs = (
            db.query(PredictionLog)
            .order_by(PredictionLog.created_at.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "prediction_id": log.id,
                "created_at": log.created_at.isoformat(),
                "final_risk_level": log.final_risk_level,
                "ml_prediction": log.ml_prediction,
                "ml_confidence": log.ml_confidence,
                "can_continue_driving": log.can_continue_driving,
                "summary": log.summary,
                "recommended_action": log.recommended_action,
                "input_data": {
                    "driving_hours_today": log.driving_hours_today,
                    "duty_window_hours": log.duty_window_hours,
                    "driving_hours_since_break": log.driving_hours_since_break,
                    "cycle_hours": log.cycle_hours,
                    "cycle_limit": log.cycle_limit,
                    "consecutive_off_duty_hours": log.consecutive_off_duty_hours,
                },
            }
            for log in logs
        ]

    finally:
        db.close()

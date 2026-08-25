import pandas as pd
import streamlit as st

from app.services.predictor import predict_hos_risk
from app.services.storage import get_recent_predictions, save_prediction


st.set_page_config(
    page_title="TruckGuard AI",
    page_icon="??",
    layout="wide",
)

st.title("?? TruckGuard AI")
st.subheader("Federal Trucking HOS Risk & Compliance Engine")

st.write(
    "Enter a driver's current Hours-of-Service clock situation. "
    "TruckGuard AI checks simplified federal HOS rules, predicts risk, "
    "explains the result, and logs the prediction."
)

st.info(
    "Educational compliance-support tool only. This does not replace an ELD, "
    "carrier safety department, FMCSA guidance, or legal compliance review."
)

with st.sidebar:
    st.header("Driver HOS Input")

    driving_hours_today = st.number_input(
        "Driving hours today",
        min_value=0.0,
        max_value=24.0,
        value=10.25,
        step=0.25,
    )

    duty_window_hours = st.number_input(
        "Hours inside 14-hour duty window",
        min_value=0.0,
        max_value=24.0,
        value=12.5,
        step=0.25,
    )

    driving_hours_since_break = st.number_input(
        "Driving hours since last qualifying break",
        min_value=0.0,
        max_value=24.0,
        value=7.25,
        step=0.25,
    )

    cycle_hours = st.number_input(
        "Cycle hours used",
        min_value=0.0,
        max_value=100.0,
        value=67.0,
        step=0.25,
    )

    cycle_limit = st.selectbox(
        "Cycle limit",
        options=[70.0, 60.0],
        index=0,
    )

    consecutive_off_duty_hours = st.number_input(
        "Consecutive off-duty hours",
        min_value=0.0,
        max_value=168.0,
        value=10.0,
        step=0.25,
    )

    run_prediction = st.button("Evaluate HOS Risk")


if run_prediction:
    payload = {
        "driving_hours_today": driving_hours_today,
        "duty_window_hours": duty_window_hours,
        "driving_hours_since_break": driving_hours_since_break,
        "cycle_hours": cycle_hours,
        "cycle_limit": cycle_limit,
        "consecutive_off_duty_hours": consecutive_off_duty_hours,
    }

    result = predict_hos_risk(payload)
    saved_record = save_prediction(result)
    result["database_log"] = saved_record

    risk_level = result["final_risk_level"]
    rules_result = result["rules_result"]
    explanation = result["explanation"]

    st.header("Risk Evaluation Result")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Final Risk Level", risk_level)
    col2.metric("ML Prediction", result["ml_prediction"])
    col3.metric("ML Confidence", f"{result['ml_confidence']:.2%}")
    col4.metric(
        "Can Continue Driving",
        "Yes" if rules_result["can_continue_driving"] else "No",
    )

    st.success(
        f"Prediction saved to database with ID: "
        f"{result['database_log']['prediction_id']}"
    )

    st.subheader("Remaining Legal Clock")
    clock_col1, clock_col2, clock_col3 = st.columns(3)

    clock_col1.metric(
        "Drive Hours Remaining",
        rules_result["remaining_drive_hours"],
    )
    clock_col2.metric(
        "Duty Window Remaining",
        rules_result["remaining_duty_window_hours"],
    )
    clock_col3.metric(
        "Cycle Hours Remaining",
        rules_result["remaining_cycle_hours"],
    )

    st.subheader("Explanation")
    st.write(explanation["summary"])

    st.subheader("Recommended Action")
    st.write(explanation["recommended_action"])

    if rules_result["blocking_reasons"]:
        st.error("Blocking Reasons")
        for reason in rules_result["blocking_reasons"]:
            st.write(f"- {reason}")

    if rules_result["warnings"]:
        st.warning("Warnings")
        for warning in rules_result["warnings"]:
            st.write(f"- {warning}")

    st.subheader("Full Prediction Output")
    st.json(result)


st.header("Recent Logged Predictions")

try:
    recent_predictions = get_recent_predictions(limit=10)

    if recent_predictions:
        recent_df = pd.DataFrame(recent_predictions)

        display_columns = [
            "prediction_id",
            "created_at",
            "final_risk_level",
            "ml_prediction",
            "ml_confidence",
            "can_continue_driving",
            "summary",
            "recommended_action",
        ]

        st.dataframe(recent_df[display_columns], hide_index=True)

    else:
        st.write("No predictions logged yet.")

except Exception as error:
    st.warning(f"Could not load recent predictions: {error}")

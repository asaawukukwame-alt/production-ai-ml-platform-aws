from pathlib import Path
import math

import joblib
import mlflow
import numpy as np
import pandas as pd

from app.services.hos_rules import HOSInput, evaluate_hos


FEATURE_COLUMNS = [
    "driving_hours_today",
    "duty_window_hours",
    "driving_hours_since_break",
    "cycle_hours",
    "cycle_limit",
    "consecutive_off_duty_hours",
]

MODEL_PATH = Path("models/hos_risk_model.pkl")


def generate_synthetic_hos_data(num_records: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    for _ in range(num_records):
        driving_hours_today = round(float(rng.uniform(0, 12.5)), 2)
        duty_window_hours = round(float(rng.uniform(0, 15.5)), 2)
        driving_hours_since_break = round(float(rng.uniform(0, 9.5)), 2)
        cycle_limit = float(rng.choice([60, 70]))
        cycle_hours = round(float(rng.uniform(0, cycle_limit + 8)), 2)
        consecutive_off_duty_hours = round(float(rng.uniform(0, 40)), 2)

        hos_input = HOSInput(
            driving_hours_today=driving_hours_today,
            duty_window_hours=duty_window_hours,
            driving_hours_since_break=driving_hours_since_break,
            cycle_hours=cycle_hours,
            cycle_limit=cycle_limit,
            consecutive_off_duty_hours=consecutive_off_duty_hours,
        )

        rule_result = evaluate_hos(hos_input)

        records.append(
            {
                "driving_hours_today": driving_hours_today,
                "duty_window_hours": duty_window_hours,
                "driving_hours_since_break": driving_hours_since_break,
                "cycle_hours": cycle_hours,
                "cycle_limit": cycle_limit,
                "consecutive_off_duty_hours": consecutive_off_duty_hours,
                "risk_level": rule_result["risk_level"],
                "can_continue_driving": rule_result["can_continue_driving"],
            }
        )

    return pd.DataFrame(records)


def bin_feature(feature_name: str, value: float, row: pd.Series) -> str:
    if feature_name == "driving_hours_today":
        if value >= 11:
            return "drive_violation"
        if value >= 9:
            return "drive_near_limit"
        if value >= 7:
            return "drive_moderate"
        return "drive_safe"

    if feature_name == "duty_window_hours":
        if value >= 14:
            return "window_violation"
        if value >= 12:
            return "window_near_limit"
        if value >= 8:
            return "window_moderate"
        return "window_safe"

    if feature_name == "driving_hours_since_break":
        if value >= 8:
            return "break_required"
        if value >= 6.5:
            return "break_approaching"
        if value >= 4:
            return "break_moderate"
        return "break_safe"

    if feature_name == "cycle_hours":
        remaining_cycle_hours = row["cycle_limit"] - value

        if remaining_cycle_hours <= 0:
            return "cycle_violation"
        if remaining_cycle_hours <= 5:
            return "cycle_near_limit"
        if remaining_cycle_hours <= 15:
            return "cycle_moderate"
        return "cycle_safe"

    if feature_name == "cycle_limit":
        return f"cycle_limit_{int(value)}"

    if feature_name == "consecutive_off_duty_hours":
        if value >= 34:
            return "restart_eligible"
        if value >= 10:
            return "full_off_duty_break"
        return "short_off_duty_break"

    return "unknown"


def make_binned_record(row: pd.Series) -> dict:
    return {
        feature: bin_feature(feature, row[feature], row)
        for feature in FEATURE_COLUMNS
    }


def stratified_split(data: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_indexes = []
    test_indexes = []

    for _, group in data.groupby("risk_level"):
        indexes = group.index.to_numpy()
        rng.shuffle(indexes)

        split_point = int(len(indexes) * (1 - test_size))

        train_indexes.extend(indexes[:split_point])
        test_indexes.extend(indexes[split_point:])

    return data.loc[train_indexes], data.loc[test_indexes]


def train_naive_bayes_classifier(train_data: pd.DataFrame) -> dict:
    labels = sorted(train_data["risk_level"].unique().tolist())
    total_rows = len(train_data)

    class_priors = {}
    conditionals = {}
    bins_by_feature = {feature: set() for feature in FEATURE_COLUMNS}

    binned_rows = []

    for _, row in train_data.iterrows():
        binned_record = make_binned_record(row)

        for feature, binned_value in binned_record.items():
            bins_by_feature[feature].add(binned_value)

        binned_rows.append(
            {
                "risk_level": row["risk_level"],
                "features": binned_record,
            }
        )

    for label in labels:
        label_rows = [
            item for item in binned_rows
            if item["risk_level"] == label
        ]

        class_priors[label] = (len(label_rows) + 1) / (total_rows + len(labels))
        conditionals[label] = {}

        for feature in FEATURE_COLUMNS:
            possible_bins = sorted(bins_by_feature[feature])
            denominator = len(label_rows) + len(possible_bins)
            conditionals[label][feature] = {}

            for possible_bin in possible_bins:
                count = sum(
                    1
                    for item in label_rows
                    if item["features"][feature] == possible_bin
                )

                conditionals[label][feature][possible_bin] = (
                    count + 1
                ) / denominator

    return {
        "model_type": "Custom Naive Bayes HOS Risk Classifier",
        "feature_columns": FEATURE_COLUMNS,
        "labels": labels,
        "class_priors": class_priors,
        "conditionals": conditionals,
        "bins_by_feature": {
            feature: sorted(values)
            for feature, values in bins_by_feature.items()
        },
    }


def predict_one(model: dict, row: pd.Series) -> tuple[str, float]:
    binned_record = make_binned_record(row)
    log_scores = {}

    for label in model["labels"]:
        log_score = math.log(model["class_priors"][label])

        for feature in model["feature_columns"]:
            binned_value = binned_record[feature]
            probability = model["conditionals"][label][feature].get(
                binned_value,
                1e-9,
            )

            log_score += math.log(probability)

        log_scores[label] = log_score

    max_log_score = max(log_scores.values())

    probabilities = {
        label: math.exp(score - max_log_score)
        for label, score in log_scores.items()
    }

    probability_total = sum(probabilities.values())

    normalized_probabilities = {
        label: probability / probability_total
        for label, probability in probabilities.items()
    }

    predicted_label = max(
        normalized_probabilities,
        key=normalized_probabilities.get,
    )

    confidence = normalized_probabilities[predicted_label]

    return predicted_label, confidence


def evaluate_model(model: dict, test_data: pd.DataFrame) -> dict:
    predictions = []
    confidences = []

    for _, row in test_data.iterrows():
        prediction, confidence = predict_one(model, row)
        predictions.append(prediction)
        confidences.append(confidence)

    actuals = test_data["risk_level"].tolist()
    correct = sum(
        1
        for actual, predicted in zip(actuals, predictions)
        if actual == predicted
    )

    accuracy = correct / len(actuals)

    labels = model["labels"]
    report_lines = []

    for label in labels:
        true_positive = sum(
            1
            for actual, predicted in zip(actuals, predictions)
            if actual == label and predicted == label
        )

        predicted_positive = sum(
            1
            for predicted in predictions
            if predicted == label
        )

        actual_positive = sum(
            1
            for actual in actuals
            if actual == label
        )

        precision = true_positive / predicted_positive if predicted_positive else 0
        recall = true_positive / actual_positive if actual_positive else 0

        report_lines.append(
            f"{label}: precision={precision:.2f}, recall={recall:.2f}, support={actual_positive}"
        )

    return {
        "accuracy": accuracy,
        "average_confidence": float(np.mean(confidences)),
        "classification_report": "\n".join(report_lines),
    }


def train_model() -> dict:
    data = generate_synthetic_hos_data()
    train_data, test_data = stratified_split(data)

    model = train_naive_bayes_classifier(train_data)
    evaluation = evaluate_model(model, test_data)

    model["training_metadata"] = {
        "num_records": len(data),
        "train_records": len(train_data),
        "test_records": len(test_data),
        "accuracy": evaluation["accuracy"],
        "average_confidence": evaluation["average_confidence"],
        "risk_distribution": data["risk_level"].value_counts().to_dict(),
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("truckguard-hos-risk")

    with mlflow.start_run():
        mlflow.log_param("model_type", model["model_type"])
        mlflow.log_param("num_records", len(data))
        mlflow.log_param("train_records", len(train_data))
        mlflow.log_param("test_records", len(test_data))

        mlflow.log_metric("accuracy", evaluation["accuracy"])
        mlflow.log_metric(
            "average_confidence",
            evaluation["average_confidence"],
        )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        mlflow.log_artifact(str(MODEL_PATH))

    return {
        "model_path": str(MODEL_PATH),
        "num_records": len(data),
        "train_records": len(train_data),
        "test_records": len(test_data),
        "accuracy": round(float(evaluation["accuracy"]), 4),
        "average_confidence": round(float(evaluation["average_confidence"]), 4),
        "risk_distribution": data["risk_level"].value_counts().to_dict(),
        "classification_report": evaluation["classification_report"],
    }


if __name__ == "__main__":
    result = train_model()

    print("TruckGuard AI ML Training Complete")
    print("=" * 60)
    print(f"Model saved to: {result['model_path']}")
    print(f"Training records: {result['num_records']}")
    print(f"Train split: {result['train_records']}")
    print(f"Test split: {result['test_records']}")
    print(f"Accuracy: {result['accuracy']}")
    print(f"Average confidence: {result['average_confidence']}")

    print()
    print("Risk Distribution:")
    for risk_level, count in result["risk_distribution"].items():
        print(f"- {risk_level}: {count}")

    print()
    print("Classification Report:")
    print(result["classification_report"])

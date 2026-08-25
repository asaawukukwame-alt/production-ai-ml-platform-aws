def build_hos_explanation(
    rules_result: dict,
    ml_prediction: str,
    ml_confidence: float,
) -> dict:
    risk_level = rules_result["risk_level"]
    can_continue = rules_result["can_continue_driving"]

    if risk_level == "HIGH":
        summary = (
            "This driver is at high HOS compliance risk and should not "
            "continue driving until the blocking issue is resolved."
        )
        recommended_action = (
            "Stop driving, resolve the listed HOS issue, and verify the "
            "driver's log before dispatching additional drive time."
        )

    elif risk_level == "MEDIUM":
        summary = (
            "This driver can continue under the simplified rule check, "
            "but is close to one or more federal HOS limits."
        )
        recommended_action = (
            "Plan a break, reduce the assignment, swap drivers, or confirm "
            "there is enough legal time before dispatching more miles."
        )

    else:
        summary = (
            "This driver appears to have available legal driving time under "
            "the simplified federal HOS rule check."
        )
        recommended_action = (
            "Continue monitoring drive time, duty window, break timing, "
            "and cycle hours before assigning additional work."
        )

    details = []

    if rules_result["blocking_reasons"]:
        details.append("Blocking reasons were found.")
        details.extend(rules_result["blocking_reasons"])

    if rules_result["warnings"]:
        details.append("Warnings were found.")
        details.extend(rules_result["warnings"])

    if not rules_result["blocking_reasons"] and not rules_result["warnings"]:
        details.append("No blocking reasons or near-limit warnings were found.")

    details.append(
        f"The ML model predicted {ml_prediction} risk with "
        f"{ml_confidence:.2%} confidence."
    )

    return {
        "summary": summary,
        "recommended_action": recommended_action,
        "details": details,
        "can_continue_driving": can_continue,
    }

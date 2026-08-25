from dataclasses import asdict, dataclass


DRIVING_LIMIT_HOURS = 11.0
DUTY_WINDOW_HOURS = 14.0
BREAK_DRIVING_HOURS = 8.0
RESTART_HOURS = 34.0


@dataclass
class HOSInput:
    driving_hours_today: float
    duty_window_hours: float
    driving_hours_since_break: float
    cycle_hours: float
    cycle_limit: float = 70.0
    consecutive_off_duty_hours: float = 0.0


@dataclass
class HOSResult:
    can_continue_driving: bool
    risk_level: str
    remaining_drive_hours: float
    remaining_duty_window_hours: float
    remaining_cycle_hours: float
    break_required: bool
    restart_eligible: bool
    blocking_reasons: list[str]
    warnings: list[str]


def evaluate_hos(data: HOSInput) -> dict:
    blocking_reasons = []
    warnings = []

    remaining_drive = max(
        0.0,
        DRIVING_LIMIT_HOURS - data.driving_hours_today,
    )

    remaining_window = max(
        0.0,
        DUTY_WINDOW_HOURS - data.duty_window_hours,
    )

    remaining_cycle = max(
        0.0,
        data.cycle_limit - data.cycle_hours,
    )

    break_required = (
        data.driving_hours_since_break >= BREAK_DRIVING_HOURS
    )

    restart_eligible = (
        data.consecutive_off_duty_hours >= RESTART_HOURS
    )

    if data.driving_hours_today >= DRIVING_LIMIT_HOURS:
        blocking_reasons.append(
            "11-hour driving limit reached."
        )

    if data.duty_window_hours >= DUTY_WINDOW_HOURS:
        blocking_reasons.append(
            "14-hour driving window reached."
        )

    if break_required:
        blocking_reasons.append(
            "30-minute qualifying break required before more driving."
        )

    if data.cycle_hours >= data.cycle_limit:
        blocking_reasons.append(
            f"{int(data.cycle_limit)}-hour cycle limit reached."
        )

    if 0 < remaining_drive <= 2:
        warnings.append(
            f"Only {remaining_drive:.1f} driving hours remain."
        )

    if 0 < remaining_window <= 2:
        warnings.append(
            f"Only {remaining_window:.1f} hours remain in the duty window."
        )

    if 0 < remaining_cycle <= 5:
        warnings.append(
            f"Only {remaining_cycle:.1f} cycle hours remain."
        )

    if 6.5 <= data.driving_hours_since_break < BREAK_DRIVING_HOURS:
        warnings.append(
            "Driver is approaching the 8-hour driving-break threshold."
        )

    can_continue = len(blocking_reasons) == 0

    if blocking_reasons:
        risk_level = "HIGH"
    elif warnings:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    result = HOSResult(
        can_continue_driving=can_continue,
        risk_level=risk_level,
        remaining_drive_hours=round(remaining_drive, 2),
        remaining_duty_window_hours=round(remaining_window, 2),
        remaining_cycle_hours=round(remaining_cycle, 2),
        break_required=break_required,
        restart_eligible=restart_eligible,
        blocking_reasons=blocking_reasons,
        warnings=warnings,
    )

    return asdict(result)

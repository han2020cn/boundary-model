from __future__ import annotations

from typing import Any, Iterable

from helpers.config import ModeAccumulator, TripRequest
from helpers.types import CandidateConstraint, Scenario


REFUSAL_PENALTY = 100.0
VEHICLE_OPERATING_COST_PER_HOUR = 40.0


def _calculate_net_expenditure(
    total_travel_distance: float,
    total_trips: int,
    served_requests: int,
    total_operating_time: float = 0.0,
    accepted_deviations: int = 0,
) -> float:
    operating_cost = (
        float(total_operating_time) / 60.0 * VEHICLE_OPERATING_COST_PER_HOUR
    )
    return float(
        total_travel_distance / 1000 * 2
        + operating_cost
        + total_trips * 15
        + 0 * accepted_deviations
        - served_requests * 3
    )

def _calculate_objective(
    total_wait: float,
    total_walk: float,
    total_onboard: float,
    total_requests: int,
    served_requests: int,
) -> float:
    timeobj = time_objective(total_wait, total_walk, total_onboard)
    refusal_requests = int(total_requests - served_requests)
    return float(
        timeobj
        + REFUSAL_PENALTY * refusal_requests
    )

def time_objective(
    total_wait: float,
    total_walk: float,
    total_onboard: float,
) -> float:
    return float(total_wait + 3* total_walk + 2* total_onboard)

def minimize_objective(
    candidates: Iterable[dict[str, Any]],
    constraints: Iterable[CandidateConstraint],
) -> tuple[dict[str, Any] | None, set[str]]:
    failure_reasons: set[str] = set()
    ordered_constraints = tuple(constraints)

    for candidate in sorted(
        candidates,
        key=lambda item: (item["objective"], item.get("ranking", ())),
    ):
        for constraint in ordered_constraints:
            failure_reason = constraint(candidate)
            if failure_reason is not None:
                failure_reasons.add(failure_reason)
                break
        else:
            return candidate, failure_reasons

    return None, failure_reasons


def init_mode_accumulator() -> ModeAccumulator:
    return ModeAccumulator()


def set_operator_metrics(
    acc: ModeAccumulator,
    total_travel_distance: float,
    total_trips: int,
    total_operating_time: float,
    accepted_deviations: int = 0,
) -> None:
    acc.total_travel_distance = float(total_travel_distance)
    acc.total_trips = int(total_trips)
    acc.operating_time = float(total_operating_time)
    acc.net_expenditure = _calculate_net_expenditure(
        acc.total_travel_distance,
        acc.total_trips,
        acc.served_requests,
        acc.operating_time,
        accepted_deviations,
    )


def finalize_nonbaseline_mode( 
    mode_id: int,
    scenario: Scenario,
    requests: list[TripRequest],
    benchmark_expenditure: float | None,
    acc: ModeAccumulator,
    feasible: bool,
    feasibility_reason: str,
) -> dict[str, Any]:
    return _finalize_result(
        mode_id=mode_id,
        scenario=scenario,
        total_requests=len(requests),
        served_requests=acc.served_requests,
        benchmark_expenditure=benchmark_expenditure,
        net_expenditure=acc.net_expenditure,
        total_wait=acc.total_wait,
        total_walk=acc.total_walk,
        total_onboard=acc.total_onboard,
        feasible=feasible,
        feasibility_reason=feasibility_reason,
        total_trips=acc.total_trips,
        operating_time=acc.operating_time,
        accepted_deviations=acc.accepted_deviations,
        max_concurrent_trips=acc.max_concurrent_trips,
        vehicle_reuse_ratio=acc.vehicle_reuse_ratio,
    )


def validate_service_policy(service_policy: str) -> str:
    if service_policy not in {"strict", "skip"}:
        raise ValueError("service_policy must be 'strict' or 'skip'")
    return service_policy


def build_expenditure_constraint(
    benchmark_expenditure: float | None,
) -> CandidateConstraint:
    def constraint(candidate: dict[str, Any]) -> str | None:
        if benchmark_expenditure is None:
            return None
        if float(candidate["candidate_expenditure"]) <= benchmark_expenditure:
            return None
        return "benchmark_exceeded"

    return constraint


def request_type_mode_reason(served_requests: int, total_requests: int) -> str:
    if served_requests < total_requests:
        return "partial_service"
    return "feasible"


def _finalize_result(
    mode_id: int,
    scenario: Scenario,
    total_requests: int,
    served_requests: int,
    benchmark_expenditure: float | None,
    net_expenditure: float,
    total_wait: float,
    total_walk: float,
    total_onboard: float,
    feasible: bool,
    feasibility_reason: str,
    total_trips: int | None = None,
    operating_time: float | None = None,
    accepted_deviations: int | None = None,
    max_concurrent_trips: int | None = None,
    vehicle_reuse_ratio: float | None = None,
) -> dict[str, Any]:
    total_service_time = total_wait + total_walk + total_onboard
    objective_value = _calculate_objective(
        total_wait,
        total_walk,
        total_onboard,
        total_requests,
        served_requests,
    )

    return {
        "scenario_id": scenario["scenario_id"],
        "lambda": scenario["lambda"],
        "ht": scenario["ht"],
        "hs": scenario["hs"],
        "seed": scenario["seed"],
        "rep_num": scenario.get("rep_num"),
        "fleet_max": int(scenario["fleet_max"]),
        "capacity": int(scenario["capacity"]),
        "mode_id": mode_id,
        "feasible": bool(feasible),
        "feasibility_reason": feasibility_reason,
        "total_requests": int(total_requests),
        "served_requests": int(served_requests),
        "unserved_requests": int(total_requests - served_requests),
        "benchmark_expenditure": _round_metric(benchmark_expenditure),
        "net_expenditure": _round_metric(net_expenditure),
        "total_wait": _round_metric(total_wait),
        "total_walk": _round_metric(total_walk),
        "total_onboard": _round_metric(total_onboard),
        "total_service_time": _round_metric(total_service_time),
        "objective_value": _round_metric(objective_value),
        "total_trips": (
            None if total_trips is None else int(total_trips)
        ),
        "operating_time": _round_metric(operating_time),
        "accepted_deviations": (
            None if accepted_deviations is None else int(accepted_deviations)
        ),
        "max_concurrent_trips": (
            None if max_concurrent_trips is None else int(max_concurrent_trips)
        ),
        "vehicle_reuse_ratio": _round_metric(vehicle_reuse_ratio),
    }


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)

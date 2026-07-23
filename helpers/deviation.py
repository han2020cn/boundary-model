from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
import math

import networkx as nx

import helpers.common_rule as rule
import helpers.fixed_loop as fixed_loop

from helpers.config import TripRequest
from helpers.types import GridNode, Scenario


def _travel_time(distance: float, fleet) -> float:
    return float(distance) / float(fleet.speed)


def _trip_key(assignment: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(assignment["loop_id"]),
        int(assignment["vehicle_id"]),
        int(assignment["departure_index"]),
    )


@dataclass
class Mode2DeviationEvent:
    trip_key: tuple[str, int, int]
    anchor_index: int
    anchor_node: GridNode
    direction: int
    depth: float
    request_ids: set[int] = field(default_factory=set)

    @property
    def key(self) -> tuple[tuple[str, int, int], int, GridNode, int]:
        return (self.trip_key, self.anchor_index, self.anchor_node, self.direction)

    @property
    def is_physical(self) -> bool:
        return self.direction != 0 and self.depth > 0.0


def _mode2_event_key(
    trip_key: tuple[str, int, int],
    anchor_index: int,
    anchor_node: GridNode,
    direction: int,
) -> tuple[tuple[str, int, int], int, GridNode, int]:
    return (trip_key, int(anchor_index), anchor_node, int(direction))


def _clone_mode2_events(
    events: dict[tuple[tuple[str, int, int], int, GridNode, int], Mode2DeviationEvent],
) -> dict[tuple[tuple[str, int, int], int, GridNode, int], Mode2DeviationEvent]:
    return {
        key: Mode2DeviationEvent(
            trip_key=event.trip_key,
            anchor_index=event.anchor_index,
            anchor_node=event.anchor_node,
            direction=event.direction,
            depth=float(event.depth),
            request_ids=set(event.request_ids),
        )
        for key, event in events.items()
    }


def _copy_trip_assignments(
    assignments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(assignment) for assignment in assignments]


def _mode2_physical_event_count(
    events: dict[tuple[tuple[str, int, int], int, GridNode, int], Mode2DeviationEvent],
) -> int:
    return sum(1 for event in events.values() if event.is_physical)


def _mode2_detour_distance(
    events: dict[tuple[tuple[str, int, int], int, GridNode, int], Mode2DeviationEvent],
) -> float:
    return sum(2.0 * float(event.depth) for event in events.values() if event.is_physical)


def _capacity_feasible_from_assignments(
    assignments: list[dict[str, Any]],
    capacity: int,
) -> bool:
    events: list[tuple[float, int]] = []
    for assignment in assignments:
        start_time = float(assignment["boarding_time"])
        end_time = float(assignment["dropoff_time"])
        if end_time <= start_time:
            return False
        events.append((start_time, 1))
        events.append((end_time, -1))

    load = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        load += delta
        if load > int(capacity):
            return False
    return True


def _grid_axis_deviation_option(nets,
    point: GridNode,
    anchor_node: GridNode,
) -> dict[str, float | int] | None:
    if not (
        isinstance(point, tuple)
        and isinstance(anchor_node, tuple)
        and len(point) == 2
        and len(anchor_node) == 2
    ):
        return None

    point_x, point_y = point
    anchor_x, anchor_y = anchor_node
    if point_x != anchor_x:
        return None

    y_distance = abs(float(point_y) - float(anchor_y))
    y_distance_m = y_distance * float(nets.grid_len)
    vehicle_deviation = min(y_distance_m, float(nets.max_dev))
    residual_walk = max(0.0, y_distance_m - float(nets.max_dev))
    if point_y > anchor_y:
        direction = 1
    elif point_y < anchor_y:
        direction = -1
    else:
        direction = 0
    return {
        "vehicle_deviation": vehicle_deviation,
        "residual_walk_d": residual_walk,
        "direction": direction,
    }


def _mode2_deviation_options(
    nets,
    point: GridNode,
    anchor_node: GridNode,
    baseline_walk_d: float,
    graph: nx.Graph,
) -> list[dict[str, float | int]]:
    options = [
        {
            "vehicle_deviation": 0.0,
            "residual_walk_d": float(baseline_walk_d),
            "direction": 0,
        }
    ]

    if getattr(nets, "_type", None) == "grid":
        deviation_option = _grid_axis_deviation_option(nets, point, anchor_node)
    else:
        distance_to_anchor = fixed_loop.distance_shortpath(point, anchor_node, graph)
        max_dev = float(getattr(nets, "max_dev", 0.0))
        deviation_option = {
            "vehicle_deviation": min(float(distance_to_anchor), max_dev),
            "residual_walk_d": max(0.0, float(distance_to_anchor) - max_dev),
            "direction": 1 if distance_to_anchor > 0.0 else 0,
        }

    if deviation_option is None:
        return options

    duplicate = any(
        math.isclose(option["vehicle_deviation"], deviation_option["vehicle_deviation"])
        and math.isclose(option["residual_walk_d"], deviation_option["residual_walk_d"])
        and int(option["direction"]) == int(deviation_option["direction"])
        for option in options
    )
    if not duplicate:
        options.append(deviation_option)
    return options


def _mode2_service_event_key(assignment: dict[str, Any], service: str):
    return assignment.get(f"{service}_event_key")


def _mode2_service_depth(assignment: dict[str, Any], service: str) -> float:
    return float(assignment.get(f"{service}_deviation_depth", 0.0))


def _mode2_residual_walk_d(assignment: dict[str, Any], service: str) -> float:
    if service == "origin":
        return float(assignment.get("origin_residual_walk_d", assignment["origin_walk_d"]))
    if service == "destination":
        return float(
            assignment.get(
                "destination_residual_walk_d",
                assignment["destination_walk_d"],
            )
        )
    raise ValueError(f"unknown Mode 2 service type: {service}")


def _mode2_anchor_for_service(
    assignment: dict[str, Any],
    service: str,
) -> tuple[int, GridNode]:
    if service == "origin":
        return int(assignment["boarding_anchor"]), assignment["boarding_node"]
    if service == "destination":
        return int(assignment["alighting_anchor"]), assignment["alighting_node"]
    raise ValueError(f"unknown Mode 2 service type: {service}")


def _apply_mode2_service_event(
    assignment: dict[str, Any],
    events: dict[tuple[tuple[str, int, int], int, GridNode, int], Mode2DeviationEvent],
    trip_key: tuple[str, int, int],
    service: str,
    option: dict[str, float | int],
) -> bool:
    depth = float(option["vehicle_deviation"])
    direction = int(option["direction"])
    if depth <= 0.0 or direction == 0:
        return False

    anchor_index, anchor_node = _mode2_anchor_for_service(assignment, service)
    key = _mode2_event_key(trip_key, anchor_index, anchor_node, direction)
    event = events.get(key)
    if event is None:
        event = Mode2DeviationEvent(
            trip_key=trip_key,
            anchor_index=anchor_index,
            anchor_node=anchor_node,
            direction=direction,
            depth=depth,
            request_ids=set(),
        )
        events[key] = event
    else:
        event.depth = max(float(event.depth), depth)
    event.request_ids.add(int(assignment["request"].request_id))

    assignment[f"{service}_event_key"] = key
    assignment[f"{service}_residual_walk_d"] = float(option["residual_walk_d"])
    assignment[f"{service}_deviation_depth"] = depth
    return True


def _simulate_trip_with_events(
    trip_assignments: list[dict[str, Any]],
    trip_events: dict[tuple[tuple[str, int, int], int, GridNode, int], Mode2DeviationEvent],
    config,
    fleet,
) -> dict[str, Any]:
    if not trip_assignments:
        return {
            "feasible": True,
            "reason": "empty_trip",
            "assignments": [],
            "total_wait": 0.0,
            "total_walk": 0.0,
            "total_onboard": 0.0,
            "objective": 0.0,
            "detour_distance": 0.0,
            "detour_time": 0.0,
            "event_count": 0,
            "final_finish_time": 0.0,
            "operating_time": 0.0,
        }

    assignments = _copy_trip_assignments(trip_assignments)
    trip_key = _trip_key(assignments[0])
    if any(_trip_key(assignment) != trip_key for assignment in assignments):
        return {
            "feasible": False,
            "reason": "mixed_trip_assignments",
            "assignments": assignments,
            "total_wait": 0.0,
            "total_walk": 0.0,
            "total_onboard": 0.0,
            "objective": math.inf,
            "detour_distance": _mode2_detour_distance(trip_events),
            "detour_time": _travel_time(_mode2_detour_distance(trip_events), fleet),
            "event_count": _mode2_physical_event_count(trip_events),
            "final_finish_time": math.inf,
            "operating_time": math.inf,
        }

    route_departure_time = float(assignments[0]["route_departure_time"])
    route_length = float(assignments[0]["route_length"])
    next_departure_time = min(
        float(assignment["next_assigned_departure_time"])
        for assignment in assignments
    )

    service_events: list[tuple[float, int, int, str]] = []
    for index, assignment in enumerate(assignments):
        pickup_position = float(assignment["boarding_offset"])
        dropoff_position = pickup_position + float(assignment["base_route_d"])
        service_events.append((pickup_position, 0, index, "origin"))
        service_events.append((dropoff_position, 1, index, "destination"))

    pickup_times: dict[int, float] = {}
    dropoff_times: dict[int, float] = {}
    accumulated_delay = 0.0
    service_events.sort(key=lambda item: (item[0], item[1], item[2]))

    cursor = 0
    while cursor < len(service_events):
        position = float(service_events[cursor][0])
        same_position: list[tuple[float, int, int, str]] = []
        while (
            cursor < len(service_events)
            and math.isclose(float(service_events[cursor][0]), position)
        ):
            same_position.append(service_events[cursor])
            cursor += 1

        anchor_arrival_time = (
            route_departure_time
            + position / float(fleet.speed)
            + accumulated_delay
        )
        anchor_departure_time = anchor_arrival_time

        deviated_services: defaultdict[
            tuple[tuple[str, int, int], int, GridNode, int],
            list[tuple[int, str]],
        ] = defaultdict(list)
        for _, _, assignment_index, service in same_position:
            assignment = assignments[assignment_index]
            event_key = _mode2_service_event_key(assignment, service)
            event = trip_events.get(event_key) if event_key is not None else None
            if event is not None and event.is_physical:
                deviated_services[event_key].append((assignment_index, service))
                continue

            if service == "destination":
                dropoff_times[assignment_index] = anchor_departure_time
                continue

            request = assignment["request"]
            ready_time = (
                float(request.departure_time)
                + _mode2_residual_walk_d(assignment, "origin") / float(config.walk_speed)
            )
            pickup_time = max(anchor_departure_time, ready_time)
            pickup_times[assignment_index] = pickup_time
            anchor_departure_time = max(anchor_departure_time, pickup_time)

        accumulated_delay += anchor_departure_time - anchor_arrival_time

        for event_key in sorted(
            deviated_services,
            key=lambda key: (key[1], fixed_loop._node_sort_key(key[2]), key[3]),
        ):
            event = trip_events[event_key]
            event_start_time = (
                route_departure_time
                + position / float(fleet.speed)
                + accumulated_delay
            )
            event_finish_time = (
                event_start_time
                + 2.0 * float(event.depth) / float(fleet.speed)
            )

            for assignment_index, service in deviated_services[event_key]:
                assignment = assignments[assignment_index]
                depth = _mode2_service_depth(assignment, service)
                point_time = event_start_time + depth / float(fleet.speed)

                if service == "destination":
                    dropoff_times[assignment_index] = point_time
                    continue

                request = assignment["request"]
                ready_time = (
                    float(request.departure_time)
                    + _mode2_residual_walk_d(assignment, "origin")
                    / float(config.walk_speed)
                )
                pickup_time = max(point_time, ready_time)
                pickup_times[assignment_index] = pickup_time
                event_finish_time = max(
                    event_finish_time,
                    pickup_time + depth / float(fleet.speed),
                )

            accumulated_delay += event_finish_time - event_start_time

    total_wait = 0.0
    total_walk = 0.0
    total_onboard = 0.0
    for index, assignment in enumerate(assignments):
        if index not in pickup_times or index not in dropoff_times:
            return {
                "feasible": False,
                "reason": "incomplete_trip_simulation",
                "assignments": assignments,
                "total_wait": 0.0,
                "total_walk": 0.0,
                "total_onboard": 0.0,
                "objective": math.inf,
                "detour_distance": _mode2_detour_distance(trip_events),
                "detour_time": _travel_time(_mode2_detour_distance(trip_events), fleet),
                "event_count": _mode2_physical_event_count(trip_events),
                "final_finish_time": math.inf,
                "operating_time": math.inf,
            }

        request = assignment["request"]
        origin_walk_d = _mode2_residual_walk_d(assignment, "origin")
        destination_walk_d = _mode2_residual_walk_d(assignment, "destination")
        pickup_ready_time = (
            float(request.departure_time)
            + origin_walk_d / float(config.walk_speed)
        )
        pickup_time = pickup_times[index]
        dropoff_time = dropoff_times[index]
        wait_time = pickup_time - pickup_ready_time
        walk_time = (origin_walk_d + destination_walk_d) / float(config.walk_speed)
        onboard_time = dropoff_time - pickup_time

        assignment["boarding_time"] = float(pickup_time)
        assignment["dropoff_time"] = float(dropoff_time)
        assignment["wait_time"] = float(wait_time)
        assignment["walk_time"] = float(walk_time)
        assignment["onboard_time"] = float(onboard_time)
        assignment["origin_walk_time"] = float(origin_walk_d) / float(config.walk_speed)
        assignment["destination_walk_time"] = (
            float(destination_walk_d) / float(config.walk_speed)
        )
        total_wait += float(wait_time)
        total_walk += float(walk_time)
        total_onboard += float(onboard_time)

    detour_distance = _mode2_detour_distance(trip_events)
    detour_time = _travel_time(detour_distance, fleet)
    final_finish_time = (
        route_departure_time
        + route_length / float(fleet.speed)
        + accumulated_delay
    )
    if final_finish_time > next_departure_time + 1e-9:
        feasible = False
        reason = "next_departure_violation"
    elif not _capacity_feasible_from_assignments(assignments, int(fleet.cap)):
        feasible = False
        reason = "capacity_limit"
    else:
        feasible = True
        reason = "feasible"

    return {
        "feasible": feasible,
        "reason": reason,
        "assignments": assignments,
        "total_wait": total_wait,
        "total_walk": total_walk,
        "total_onboard": total_onboard,
        "objective": rule.time_objective(total_wait, total_walk, total_onboard),
        "detour_distance": detour_distance,
        "detour_time": detour_time,
        "event_count": _mode2_physical_event_count(trip_events),
        "final_finish_time": final_finish_time,
        "operating_time": final_finish_time - route_departure_time,
    }


def evaluate_2(
    config,
    nets,
    fleet,
    baseline: dict[str, Any],
    requests: list[TripRequest],
    scenario: Scenario,
    network_context,
    benchmark_expenditure: float | None,
    service_policy: str = "strict",
) -> dict[str, Any]:
    service_policy = rule.validate_service_policy(service_policy)
    acc = rule.init_mode_accumulator()
    baseline_result = baseline["result"]
    assignments = list(baseline["assignments"])
    fixed_metrics = baseline["fixed_metrics"]
    if not bool(baseline_result["feasible"]):
        return rule.finalize_nonbaseline_mode(
            mode_id=2,
            scenario=scenario,
            requests=requests,
            benchmark_expenditure=benchmark_expenditure,
            acc=acc,
            feasible=False,
            feasibility_reason="mode1_baseline_infeasible",
        )

    assignments_by_trip: defaultdict[
        tuple[str, int, int],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for assignment in assignments:
        assignments_by_trip[_trip_key(assignment)].append(assignment)

    final_assignments: list[dict[str, Any]] = []
    final_trip_events: dict[
        tuple[str, int, int],
        dict[tuple[tuple[str, int, int], int, GridNode, int], Mode2DeviationEvent],
    ] = {}
    accepted_detour_distance = 0.0
    accepted_extra_operating_time = 0.0

    for trip_key in sorted(assignments_by_trip):
        current_trip_assignments = _copy_trip_assignments(assignments_by_trip[trip_key])
        current_trip_events: dict[
            tuple[tuple[str, int, int], int, GridNode, int],
            Mode2DeviationEvent,
        ] = {}
        current_trip_metrics = _simulate_trip_with_events(
            current_trip_assignments,
            current_trip_events,
            config,
            fleet,
        )
        if current_trip_metrics["feasible"]:
            current_trip_assignments = current_trip_metrics["assignments"]
        else:
            current_trip_metrics = {
                "feasible": True,
                "reason": "baseline_metrics_used",
                "assignments": current_trip_assignments,
                "total_wait": sum(
                    float(item["wait_time"]) for item in current_trip_assignments
                ),
                "total_walk": sum(
                    float(item["walk_time"]) for item in current_trip_assignments
                ),
                "total_onboard": sum(
                    float(item["onboard_time"]) for item in current_trip_assignments
                ),
                "objective": rule.time_objective(
                    sum(float(item["wait_time"]) for item in current_trip_assignments),
                    sum(float(item["walk_time"]) for item in current_trip_assignments),
                    sum(float(item["onboard_time"]) for item in current_trip_assignments),
                ),
                "detour_distance": 0.0,
                "detour_time": 0.0,
                "event_count": 0,
                "final_finish_time": (
                    float(current_trip_assignments[0]["route_departure_time"])
                    + float(current_trip_assignments[0]["route_length"])
                    / float(fleet.speed)
                ),
                "operating_time": (
                    float(current_trip_assignments[0]["route_length"])
                    / float(fleet.speed)
                ),
            }

        request_order = sorted(
            range(len(current_trip_assignments)),
            key=lambda index: (
                -float(current_trip_assignments[index]["walk_time"]),
                float(current_trip_assignments[index]["request"].departure_time),
                int(current_trip_assignments[index]["request"].request_id),
            ),
        )

        for assignment_index in request_order:
            assignment = current_trip_assignments[assignment_index]
            request = assignment["request"]
            origin_options = [
                option
                for option in _mode2_deviation_options(
                    nets,
                    request.origin,
                    assignment["boarding_node"],
                    float(assignment["origin_walk_d"]),
                    network_context.graph,
                )
                if float(option["vehicle_deviation"]) > 0.0
                and int(option["direction"]) != 0
            ]
            destination_options = [
                option
                for option in _mode2_deviation_options(
                    nets,
                    request.destination,
                    assignment["alighting_node"],
                    float(assignment["destination_walk_d"]),
                    network_context.graph,
                )
                if float(option["vehicle_deviation"]) > 0.0
                and int(option["direction"]) != 0
            ]

            proposals: list[dict[str, dict[str, float | int]]] = []
            proposals.extend({"origin": option} for option in origin_options)
            proposals.extend({"destination": option} for option in destination_options)
            proposals.extend(
                {"origin": origin_option, "destination": destination_option}
                for origin_option in origin_options
                for destination_option in destination_options
            )

            candidates: list[dict[str, Any]] = []
            for proposal in proposals:
                candidate_assignments = _copy_trip_assignments(current_trip_assignments)
                candidate_events = _clone_mode2_events(current_trip_events)
                candidate_assignment = candidate_assignments[assignment_index]
                applied = False
                for service, option in proposal.items():
                    applied = (
                        _apply_mode2_service_event(
                            candidate_assignment,
                            candidate_events,
                            trip_key,
                            service,
                            option,
                        )
                        or applied
                    )
                if not applied:
                    continue

                candidate_metrics = _simulate_trip_with_events(
                    candidate_assignments,
                    candidate_events,
                    config,
                    fleet,
                )
                if not candidate_metrics["feasible"]:
                    continue
                if (
                    float(candidate_metrics["objective"])
                    >= float(current_trip_metrics["objective"]) - 1e-9
                ):
                    continue
                base_trip_operating_time = (
                    float(candidate_assignments[0]["route_length"])
                    / float(fleet.speed)
                )
                candidate_extra_operating_time = max(
                    0.0,
                    float(candidate_metrics["operating_time"])
                    - base_trip_operating_time,
                )
                candidate_net_expenditure = rule._calculate_net_expenditure(
                    float(fixed_metrics["total_travel_distance"])
                    + accepted_detour_distance
                    + _mode2_detour_distance(candidate_events),
                    int(fixed_metrics["total_trips"]),
                    len(assignments),
                    float(fixed_metrics["operating_time"])
                    + accepted_extra_operating_time
                    + candidate_extra_operating_time,
                    _mode2_physical_event_count(candidate_events),
                )
                if (
                    benchmark_expenditure is not None
                    and candidate_net_expenditure
                    > float(benchmark_expenditure) + 1e-9
                ):
                    continue

                candidates.append(
                    {
                        "metrics": candidate_metrics,
                        "events": candidate_events,
                        "ranking": (
                            float(candidate_metrics["objective"]),
                            float(candidate_metrics["detour_distance"]),
                            int(candidate_metrics["event_count"]),
                        ),
                    }
                )

            if not candidates:
                continue

            best_choice = min(candidates, key=lambda item: item["ranking"])
            current_trip_events = best_choice["events"]
            current_trip_metrics = best_choice["metrics"]
            current_trip_assignments = current_trip_metrics["assignments"]

        final_assignments.extend(current_trip_assignments)
        final_trip_events[trip_key] = current_trip_events
        accepted_detour_distance += _mode2_detour_distance(current_trip_events)
        base_trip_operating_time = (
            float(current_trip_assignments[0]["route_length"])
            / float(fleet.speed)
        )
        accepted_extra_operating_time += max(
            0.0,
            float(current_trip_metrics["operating_time"])
            - base_trip_operating_time,
        )

    acc.served_requests = len(final_assignments)
    acc.total_wait = sum(float(item["wait_time"]) for item in final_assignments)
    acc.total_walk = sum(float(item["walk_time"]) for item in final_assignments)
    acc.total_onboard = sum(float(item["onboard_time"]) for item in final_assignments)
    total_mode2_detour_distance = accepted_detour_distance
    total_mode2_operating_time = (
        float(fixed_metrics["operating_time"])
        + accepted_extra_operating_time
    )
    acc.accepted_deviations = sum(
        _mode2_physical_event_count(events)
        for events in final_trip_events.values()
    )
    rule.set_operator_metrics(
        acc,
        float(fixed_metrics["total_travel_distance"]) + total_mode2_detour_distance,
        int(fixed_metrics["total_trips"]),
        total_mode2_operating_time,
        acc.accepted_deviations,
    )

    within_budget = (
        benchmark_expenditure is None
        or acc.net_expenditure <= float(benchmark_expenditure) + 1e-9
    )
    return rule.finalize_nonbaseline_mode(
        mode_id=2,
        scenario=scenario,
        requests=requests,
        benchmark_expenditure=benchmark_expenditure,
        acc=acc,
        feasible=within_budget,
        feasibility_reason=(
            rule.request_type_mode_reason(acc.served_requests, len(requests))
            if within_budget
            else "benchmark_exceeded"
        ),
    )

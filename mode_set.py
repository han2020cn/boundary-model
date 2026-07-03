from __future__ import annotations

from collections import defaultdict
from typing import Any
import math
import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
import helpers.common_rule as rule
import helpers.drt as drt 
import helpers.fixed_loop as fixed_loop
import helpers.fixedstep as fix
import helpers.functions as fs

from helpers.config import LoopContext, SpokeVehicle, TripRequest
from helpers.types import GridNode, Scenario


def _travel_time(distance: float, fleet) -> float:
    return float(distance) / float(fleet.speed)


def _build_mode1_baseline(
    config,
    nets,
    fleet,
    requests: list[TripRequest],
    scenario: Scenario,
    network_context,
) -> dict[str, Any]:
    walk_v = config.walk_speed
    bus_v = fleet.speed
    loops, weighted_contexts, fixed_metrics = fix.build_context(network_context,config, nets, fleet)
    if not fixed_metrics["feasible"]:
        raise ValueError("mode 1 is not feasible")

    loads: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    served_requests = 0
    all_wait = 0.0
    all_walk = 0.0
    all_onboard = 0.0
    assignments: list[dict[str, Any]] = []

    for request in _sorted_requests(requests):
        loop = fixed_loop._nearest_loop_for_request(request, loops, network_context.graph)
        loop_context_choice = weighted_contexts[loop.id]
        stop_offsets_d = loop_context_choice["offsets"]
        route_length = float(loop_context_choice["length"])
        origin_walk_d, boarding_stop, boarding_index = min(
            (
                (
                    fixed_loop.distance_shortpath(request.origin, stop, network_context.graph),
                    stop,
                    index,
                )
                for stop, index in loop.fixed_stop_indices.items()
            ),
            key=lambda item: (item[0], fixed_loop._node_sort_key(item[1])),
        )
        destination_walk_d, alighting_stop, alighting_index = min(
            (
                (
                    fixed_loop.distance_shortpath(request.destination, stop, network_context.graph),
                    stop,
                    index,
                )
                for stop, index in loop.fixed_stop_indices.items()
            ),
            key=lambda item: (item[0], fixed_loop._node_sort_key(item[1])),
        )
        walk_t = float(origin_walk_d + destination_walk_d) / walk_v
        boarding_offset_d = float(stop_offsets_d[boarding_stop])
        alighting_offset_d = float(stop_offsets_d[alighting_stop])
        base_route_d = fixed_loop._calculate_travel_weighted(
            boarding_offset_d,
            alighting_offset_d,
            route_length,
        )
        onboard_t = bus_v * float(base_route_d)
        earliest_boarding_time = float(request.departure_time) + float(origin_walk_d / walk_v)
        scheduled_passes = fixed_loop._scheduled_pass_candidates(
            config,
            fleet,
            loop,
            boarding_offset_d,
            earliest_boarding_time,
        )
        best_choice = None
        for scheduled_pass in scheduled_passes:
            vehicle_id = int(scheduled_pass["vehicle_id"])
            boarding_time = float(scheduled_pass["pass_time"])
            wait_time = float(boarding_time - earliest_boarding_time)

            if not fixed_loop._check_loop_capacity(
                fleet,
                loads,
                vehicle_id,
                boarding_time,
                boarding_index,
                alighting_index,
                loop.length,
            ):
                continue

            route_travel_time = _travel_time(base_route_d, fleet)
            best_choice = {
                "request": request,
                "loop": loop,
                "loop_id": loop.id,
                "vehicle_id": vehicle_id,
                "departure_index": int(scheduled_pass["departure_index"]),
                "route_departure_time": float(scheduled_pass["route_departure_time"]),
                "boarding_node": boarding_stop,
                "alighting_node": alighting_stop,
                "boarding_anchor": int(boarding_index),
                "alighting_anchor": int(alighting_index),
                "boarding_offset": boarding_offset_d,
                "alighting_offset": alighting_offset_d,
                "route_length": route_length,
                "base_route_d": float(base_route_d),
                "boarding_time": boarding_time,
                "dropoff_time": boarding_time + route_travel_time,
                "wait_time": wait_time,
                "walk_time": walk_t,
                "onboard_time": onboard_t,
                "origin_walk_d": float(origin_walk_d),
                "destination_walk_d": float(destination_walk_d),
                "origin_walk_time": float(origin_walk_d) / walk_v,
                "destination_walk_time": float(destination_walk_d) / walk_v,
                "next_assigned_departure_time": fixed_loop._next_assigned_departure_time(
                    config,
                    fleet,
                    loop,
                    int(scheduled_pass["departure_index"]),
                ),
            }
            break

        if best_choice is None:
            result = rule._finalize_result(
                mode_id=1,
                scenario=scenario,
                total_requests=len(requests),
                served_requests=served_requests,
                benchmark_expenditure=None,
                net_expenditure=rule._calculate_net_expenditure(
                    float(fixed_metrics["total_travel_distance"]),
                    int(fixed_metrics["total_trips"]),
                    served_requests,
                ),
                total_wait=all_wait,
                total_walk=all_walk,
                total_onboard=all_onboard,
                feasible=False,
                feasibility_reason="",
            )
            result["benchmark_expenditure"] = None
            return {
                "result": result,
                "assignments": assignments,
                "loops": loops,
                "weighted_contexts": weighted_contexts,
                "fixed_metrics": fixed_metrics,
            }

        fixed_loop._reserve_loop_capacity(
            loads,
            best_choice["vehicle_id"],
            best_choice["boarding_time"],
            best_choice["boarding_anchor"],
            best_choice["alighting_anchor"],
            loop.length,
        )
        assignments.append(best_choice)
        served_requests += 1
        all_wait += float(best_choice["wait_time"])
        all_walk += float(best_choice["walk_time"])
        all_onboard += float(best_choice["onboard_time"])

    net_expenditure = rule._calculate_net_expenditure(
        float(fixed_metrics["total_travel_distance"]),
        int(fixed_metrics["total_trips"]),
        served_requests,
    )
    result = rule._finalize_result(
        mode_id=1,
        scenario=scenario,
        total_requests=len(requests),
        served_requests=served_requests,
        benchmark_expenditure=net_expenditure,
        net_expenditure=net_expenditure,
        total_wait=all_wait,
        total_walk=all_walk,
        total_onboard=all_onboard,
        feasible=True,
        feasibility_reason="feasible",
    )
    result["benchmark_expenditure"] = result["net_expenditure"]
    return {
        "result": result,
        "assignments": assignments,
        "loops": loops,
        "weighted_contexts": weighted_contexts,
        "fixed_metrics": fixed_metrics,
    }

# 评估固定路线模式
def evaluate_1( 
    config, nets, fleet,
    requests: list[TripRequest],
    scenario: Scenario,
    network_context,
) -> dict[str, Any]:
    return _build_mode1_baseline(
        config,
        nets,
        fleet,
        requests,
        scenario,
        network_context,
    )["result"]


def _trip_key(assignment: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(assignment["loop_id"]),
        int(assignment["vehicle_id"]),
        int(assignment["departure_index"]),
    )


def _seat_interval_capacity_feasible(
    intervals: list[dict[str, Any]],
    request_id: int,
    start_time: float,
    end_time: float,
    capacity: int,
) -> bool:
    if end_time <= start_time:
        return False

    events: list[tuple[float, int]] = [(float(start_time), 1), (float(end_time), -1)]
    for interval in intervals:
        if int(interval["request_id"]) == int(request_id):
            continue
        existing_start = float(interval["start"])
        existing_end = float(interval["end"])
        if existing_start < end_time and start_time < existing_end:
            events.append((existing_start, 1))
            events.append((existing_end, -1))

    load = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        load += delta
        if load > int(capacity):
            return False
    return True


def _grid_axis_deviation_option(
    point: GridNode,
    anchor_node: GridNode,
) -> dict[str, float] | None:
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
    vehicle_deviation = min(y_distance, 2.0)
    residual_walk = max(0.0, y_distance - 2.0)
    return {
        "vehicle_deviation": vehicle_deviation,
        "residual_walk_d": residual_walk,
    }


def _mode2_deviation_options(
    nets,
    point: GridNode,
    anchor_node: GridNode,
    baseline_walk_d: float,
    graph: nx.Graph,
) -> list[dict[str, float]]:
    options = [
        {
            "vehicle_deviation": 0.0,
            "residual_walk_d": float(baseline_walk_d),
        }
    ]

    if getattr(nets, "_type", None) == "grid":
        deviation_option = _grid_axis_deviation_option(point, anchor_node)
    else:
        distance_to_anchor = fixed_loop.distance_shortpath(point, anchor_node, graph)
        max_dev = float(getattr(nets, "max_dev", 0.0))
        deviation_option = {
            "vehicle_deviation": min(float(distance_to_anchor), max_dev),
            "residual_walk_d": max(0.0, float(distance_to_anchor) - max_dev),
        }

    if deviation_option is None:
        return options

    duplicate = any(
        math.isclose(option["vehicle_deviation"], deviation_option["vehicle_deviation"])
        and math.isclose(option["residual_walk_d"], deviation_option["residual_walk_d"])
        for option in options
    )
    if not duplicate:
        options.append(deviation_option)
    return options


def _update_interval(
    intervals: list[dict[str, Any]],
    request_id: int,
    start_time: float,
    end_time: float,
) -> None:
    for interval in intervals:
        if int(interval["request_id"]) == int(request_id):
            interval["start"] = float(start_time)
            interval["end"] = float(end_time)
            return
    intervals.append(
        {
            "request_id": int(request_id),
            "start": float(start_time),
            "end": float(end_time),
        }
    )

# 评估偏离路线模式 deviated route
def evaluate_2( 
    config, nets, fleet,
    requests: list[TripRequest],
    scenario: Scenario,
    network_context,
    benchmark_expenditure: float | None,
    service_policy: str = "strict",
) -> dict[str, Any]:
    service_policy = rule.validate_service_policy(service_policy)
    walk_v = config.walk_speed
    acc = rule.init_mode_accumulator() # 初始化一个ModeAccumulator对象acc，用于累计评估指标

    try:
        baseline = _build_mode1_baseline(
            config,
            nets,
            fleet,
            requests,
            scenario,
            network_context,
        )
    except ValueError as exc:
        return rule.finalize_nonbaseline_mode( #
            mode_id=2,
            scenario=scenario,
            requests=requests,
            benchmark_expenditure=benchmark_expenditure,
            acc=acc,
            feasible=False,
            feasibility_reason=str(exc),
        )

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

    acc.served_requests = len(assignments)
    acc.total_wait = sum(float(item["wait_time"]) for item in assignments)
    acc.total_walk = sum(float(item["walk_time"]) for item in assignments)
    acc.total_onboard = sum(float(item["onboard_time"]) for item in assignments)
    rule.set_operator_metrics(
        acc,
        float(fixed_metrics["total_travel_distance"]),
        int(fixed_metrics["total_trips"]),
    )

    seat_intervals: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    trip_detour_time: defaultdict[tuple[str, int, int], float] = defaultdict(float)
    trip_detour_distance: defaultdict[tuple[str, int, int], float] = defaultdict(float)
    for assignment in assignments:
        vehicle_id = int(assignment["vehicle_id"])
        seat_intervals[vehicle_id].append(
            {
                "request_id": int(assignment["request"].request_id),
                "start": float(assignment["boarding_time"]),
                "end": float(assignment["dropoff_time"]),
            }
        )

    prioritized_assignments = sorted(
        assignments,
        key=lambda item: (
            -float(item["walk_time"]),
            float(item["request"].departure_time),
            int(item["request"].request_id),
        ),
    )

    for assignment in prioritized_assignments:
        request = assignment["request"]
        vehicle_id = int(assignment["vehicle_id"])
        trip_key = _trip_key(assignment)
        origin_options = _mode2_deviation_options(
            nets,
            request.origin,
            assignment["boarding_node"],
            float(assignment["origin_walk_d"]),
            network_context.graph,
        )
        destination_options = _mode2_deviation_options(
            nets,
            request.destination,
            assignment["alighting_node"],
            float(assignment["destination_walk_d"]),
            network_context.graph,
        )
        candidates: list[dict[str, Any]] = []

        for origin_option in origin_options:
            for destination_option in destination_options:
                origin_deviation_d = float(origin_option["vehicle_deviation"])
                destination_deviation_d = float(destination_option["vehicle_deviation"])
                origin_walk_d = float(origin_option["residual_walk_d"])
                destination_walk_d = float(destination_option["residual_walk_d"])
                detour_distance = 2.0 * (origin_deviation_d + destination_deviation_d)
                detour_time = _travel_time(detour_distance, fleet)

                planned_anchor_time = (
                    float(assignment["boarding_time"])
                    + float(trip_detour_time[trip_key])
                )
                vehicle_arrival_at_pickup = (
                    planned_anchor_time
                    + _travel_time(origin_deviation_d, fleet)
                )
                pickup_ready_time = (
                    float(request.departure_time)
                    + float(origin_walk_d) / float(walk_v)
                )
                pickup_time = max(vehicle_arrival_at_pickup, pickup_ready_time)
                route_resume_time = (
                    pickup_time
                    + _travel_time(origin_deviation_d, fleet)
                )
                alighting_anchor_time = (
                    route_resume_time
                    + _travel_time(float(assignment["base_route_d"]), fleet)
                )
                dropoff_time = (
                    alighting_anchor_time
                    + _travel_time(destination_deviation_d, fleet)
                )
                finish_time = (
                    dropoff_time
                    + _travel_time(destination_deviation_d, fleet)
                )
                walk_time = float(origin_walk_d + destination_walk_d) / float(walk_v)
                wait_time = pickup_time - pickup_ready_time
                onboard_time = dropoff_time - pickup_time
                delta_wait = wait_time - float(assignment["wait_time"])
                delta_walk = walk_time - float(assignment["walk_time"])
                delta_onboard = onboard_time - float(assignment["onboard_time"])
                delta_objective = 2.0 * delta_wait + 3.0 * delta_walk + delta_onboard

                next_departure_time = float(assignment["next_assigned_departure_time"])
                trip_finish_time = (
                    float(assignment["route_departure_time"])
                    + _travel_time(float(assignment["route_length"]), fleet)
                    + float(trip_detour_time[trip_key])
                    + detour_time
                )
                if finish_time > next_departure_time + 1e-9:
                    continue
                if trip_finish_time > next_departure_time + 1e-9:
                    continue
                if pickup_time < pickup_ready_time - 1e-9:
                    continue
                if not _seat_interval_capacity_feasible(
                    seat_intervals[vehicle_id],
                    int(request.request_id),
                    pickup_time,
                    dropoff_time,
                    int(fleet.cap),
                ):
                    continue

                candidates.append(
                    {
                        "delta_objective": delta_objective,
                        "delta_wait": delta_wait,
                        "delta_walk": delta_walk,
                        "delta_onboard": delta_onboard,
                        "wait_time": wait_time,
                        "walk_time": walk_time,
                        "onboard_time": onboard_time,
                        "pickup_time": pickup_time,
                        "dropoff_time": dropoff_time,
                        "detour_distance": detour_distance,
                        "detour_time": detour_time,
                        "ranking": (
                            delta_objective,
                            wait_time + walk_time + onboard_time,
                            detour_distance,
                        ),
                    }
                )

        if not candidates:
            continue

        best_choice = min(candidates, key=lambda item: item["ranking"])
        if float(best_choice["delta_objective"]) >= -1e-9:
            continue

        acc.total_wait += float(best_choice["delta_wait"])
        acc.total_walk += float(best_choice["delta_walk"])
        acc.total_onboard += float(best_choice["delta_onboard"])
        trip_detour_time[trip_key] += float(best_choice["detour_time"])
        trip_detour_distance[trip_key] += float(best_choice["detour_distance"])
        _update_interval(
            seat_intervals[vehicle_id],
            int(request.request_id),
            float(best_choice["pickup_time"]),
            float(best_choice["dropoff_time"]),
        )
        rule.set_operator_metrics(
            acc,
            float(fixed_metrics["total_travel_distance"])
            + sum(trip_detour_distance.values()),
            int(fixed_metrics["total_trips"]),
        )

    if service_policy == "strict" and acc.served_requests < len(requests):
        return rule.finalize_nonbaseline_mode(
            mode_id=2,
            scenario=scenario,
            requests=requests,
            benchmark_expenditure=benchmark_expenditure,
            acc=acc,
            feasible=False,
            feasibility_reason="mode1_baseline_incomplete",
            )

    return rule.finalize_nonbaseline_mode(
        mode_id=2,
        scenario=scenario,
        requests=requests,
        benchmark_expenditure=benchmark_expenditure,
        acc=acc,
        feasible=True,
        feasibility_reason=rule.request_type_mode_reason(acc.served_requests, len(requests)),
    )


def evaluate_3( # 评估动态路线模式 DRT with clustered pre-booking（聚类预订）
    config,
    nets,
    fleet,
    requests: list[TripRequest],
    scenario: Scenario,
    network_context,
    benchmark_expenditure: float | None,
    service_policy: str = "strict",
) -> dict[str, Any]:

    service_policy = rule.validate_service_policy(service_policy)
    acc = rule.init_mode_accumulator()
    prebooking_requests, realtime_requests = requests_by_type(requests)
    hub = network_context.hub
    vehicle_states: dict[int, drt.DrtVehicleState] = {
        vehicle_id: drt.DrtVehicleState(current_location=hub)
        for vehicle_id in range(fleet.num)
    }

    for state in vehicle_states.values():
        state.pending_evaluation = drt._evaluate_drt_event_schedule(
            [],
            network_context.graph,
            nets,
            fleet,
            start_location=hub,
        )

    scheduled_request_ids: set[int] = set()
    skipped_request_ids: set[int] = set()
    all_vehicle_ids = tuple(vehicle_states)
    next_trip_id = 1
    if not all_vehicle_ids:
        return rule.finalize_nonbaseline_mode(
            mode_id=3,
            scenario=scenario,
            requests=requests,
            benchmark_expenditure=benchmark_expenditure,
            acc=acc,
            feasible=False,
            feasibility_reason="no_vehicle",
        )

    if fleet.num == 1:
        prebooking_vehicle_count = 1
    else:
        prebooking_vehicle_count = max(1, min(fleet.num - 1, math.floor(0.8 * fleet.num)))
    prebooking_vehicle_ids = all_vehicle_ids[:prebooking_vehicle_count]
    reserved_vehicle_ids = all_vehicle_ids[prebooking_vehicle_count:] or all_vehicle_ids

    def sync_accumulator() -> None: # 同步accumulator的指标，基于当前车辆状态和已安排的请求
        totals = drt._drt_state_totals(vehicle_states)
        trips = drt._drt_trips_by_state(vehicle_states)
        acc.served_requests = len(scheduled_request_ids)
        acc.total_wait = totals["wait"]
        acc.total_walk = 0.0
        acc.total_onboard = totals["onboard"]
        rule.set_operator_metrics(
            acc,
            totals["travel_distance"],
            int(totals["trip"]),
        )
        acc.total_trips = int(totals["trip"])
        acc.max_concurrent_trips = drt._max_concurrent_trips(trips)
        acc.vehicle_reuse_ratio = (
            None
            if fleet.num <= 0
            else float(acc.total_trips) / float(fleet.num)
        )

    #TODO
    def _insert_request_lowest_cost( # 尝试将请求插入到允许的车辆中，选择增加的travel distance（行驶距离）最小的插入方式，如果成功插入返回True，否则返回False
        request: TripRequest,
        allowed_vehicle_ids: tuple[int, ...],
        *,
        use_pickup_filter: bool = True,
    ) -> bool:
        nonlocal next_trip_id
        allowed_vehicle_ids = tuple(
            vehicle_id for vehicle_id in allowed_vehicle_ids if vehicle_id in vehicle_states
        )
        if not allowed_vehicle_ids:
            return False

        current_totals = drt._drt_state_totals(vehicle_states)
        current_total_wait = float(current_totals["wait"])
        current_total_onboard = float(current_totals["onboard"])
        current_total_travel = float(current_totals["travel_distance"])
        pickup_event = drt.DrtEvent(request=request, event_type="pickup")
        dropoff_event = drt.DrtEvent(request=request, event_type="dropoff")

        state_metrics: dict[int, dict[str, float]] = {}
        for vehicle_id in allowed_vehicle_ids:
            state = vehicle_states[vehicle_id]
            state_metrics[vehicle_id] = {
                "pending_wait": drt._sum_evaluation_metric(
                    state.pending_evaluation,
                    "wait",
                ),
                "pending_onboard": drt._sum_evaluation_metric(
                    state.pending_evaluation,
                    "onboard",
                ),
                "pending_travel": float(state.pending_evaluation["active_travel"]),
                "workload": float(
                    len(state.pending_events)
                    + state.pending_evaluation.get("active_travel", 0.0)
                ),
            }
# TODO
        def pickup_candidates() -> list[tuple[int, int, float, float, float]]: # 生成候选的pickup插入位置，基于允许的车辆ID和当前车辆状态，评估每个插入位置的pickup delay（乘客等待时间）、origin proximity（起点接近度）和workload（工作负载），返回排序后的候选列表
            candidates: list[tuple[int, int, float, float, float]] = []
            for vehicle_id in allowed_vehicle_ids:
                state = vehicle_states[vehicle_id]
                schedule = state.pending_events
                for pickup_position in range(len(schedule) + 1):
                    schedule_with_pickup = (
                        schedule[:pickup_position]
                        + [pickup_event]
                        + schedule[pickup_position:]
                    )
                    pickup_evaluation = drt._evaluate_drt_event_schedule(
                        schedule_with_pickup,
                        network_context.graph,
                        nets,
                        fleet,
                        start_time=state.current_time,
                        start_location=state.current_location,
                        onboard_requests=state.onboard_requests,
                        onboard_pickup_times=state.onboard_pickup_times,
                    )
                    pickup_record = next(
                        (
                            record
                            for record in pickup_evaluation["events"]
                            if record["event"] == pickup_event
                        ),
                        None,
                    )
                    if pickup_record is None:
                        continue

                    pickup_delay = max(
                        0.0,
                        float(pickup_record["event_time"]) - request.departure_time,
                    )
                    origin_proximity = float(
                        pickup_record["active_travel_increment"]
                    )
                    workload = state_metrics[vehicle_id]["workload"]
                    candidates.append(
                        (
                            vehicle_id,
                            pickup_position,
                            pickup_delay,
                            origin_proximity,
                            workload,
                        )
                    )

            candidates.sort(
                key=lambda item: (
                    item[2],
                    item[3],
                    item[4],
                    item[0],
                    item[1],
                )
            )
            limit = int(
                getattr(
                    config,
                    "drt_pickup_candidate_limit",
                    max(20, 4 * len(allowed_vehicle_ids)),
                )
            )
            return candidates[: max(1, limit)]

        if use_pickup_filter:
            pickup_pairs = {
                (vehicle_id, pickup_position)
                for vehicle_id, pickup_position, *_ in pickup_candidates()
            }
        else:
            pickup_pairs = {
                (vehicle_id, pickup_position)
                for vehicle_id in allowed_vehicle_ids
                for pickup_position in range(
                    len(vehicle_states[vehicle_id].pending_events) + 1
                )
            }

        candidates: list[dict[str, Any]] = []
        for vehicle_id, pickup_position in sorted(pickup_pairs):
            state = vehicle_states[vehicle_id]
            schedule = state.pending_events
            schedule_with_pickup = (
                schedule[:pickup_position]
                + [pickup_event]
                + schedule[pickup_position:]
            )
            for dropoff_position in range(
                pickup_position + 1,
                len(schedule_with_pickup) + 1,
            ):
                candidate_schedule = (
                    schedule_with_pickup[:dropoff_position]
                    + [dropoff_event]
                    + schedule_with_pickup[dropoff_position:]
                )
                candidate_evaluation = drt._evaluate_drt_event_schedule(
                    candidate_schedule,
                    network_context.graph,
                    nets,
                    fleet,
                    start_time=state.current_time,
                    start_location=state.current_location,
                    onboard_requests=state.onboard_requests,
                    onboard_pickup_times=state.onboard_pickup_times,
                )
                metrics = state_metrics[vehicle_id]
                candidate_pending_wait = drt._sum_evaluation_metric(
                    candidate_evaluation,
                    "wait",
                )
                candidate_pending_onboard = drt._sum_evaluation_metric(
                    candidate_evaluation,
                    "onboard",
                )
                candidate_pending_travel = float(candidate_evaluation["active_travel"])
                delta_wait = candidate_pending_wait - metrics["pending_wait"]
                delta_onboard = (
                    candidate_pending_onboard - metrics["pending_onboard"]
                )
                delta_travel = candidate_pending_travel - metrics["pending_travel"]
                candidate_total_wait = current_total_wait + delta_wait
                candidate_total_onboard = current_total_onboard + delta_onboard
                candidate_total_travel = current_total_travel + delta_travel
                candidate_departures = int(
                    current_totals["trip"]
                    + int(not drt._state_has_active_trip(state))
                )
                candidate_expenditure = rule._calculate_net_expenditure(
                    candidate_total_travel,
                    candidate_departures,
                    len(scheduled_request_ids) + 1,
                )
                pickup_record = next(
                    (
                        record
                        for record in candidate_evaluation["events"]
                        if record["event"] == pickup_event
                    ),
                    None,
                )
                pickup_delay = (
                    max(
                        0.0,
                        float(pickup_record["event_time"]) - request.departure_time,
                    )
                    if pickup_record is not None
                    else float("inf")
                )
                insertion_cost = (
                    delta_wait
                    + delta_onboard
                    + delta_travel
                )
                ranking = (
                    pickup_delay,
                    delta_wait,
                    delta_onboard,
                    delta_travel,
                    candidate_total_wait,
                    candidate_total_onboard,
                    candidate_expenditure,
                    vehicle_id,
                    pickup_position,
                    dropoff_position,
                )
                candidates.append(
                    {
                        "vehicle_id": vehicle_id,
                        "schedule": candidate_schedule,
                        "evaluation": candidate_evaluation,
                        "candidate_departures": candidate_departures,
                        "candidate_expenditure": candidate_expenditure,
                        "objective": insertion_cost,
                        "ranking": ranking,
                    }
                )

        best_insertion, _ = rule.minimize_objective(
            candidates,
            (
                drt._build_drt_capacity_constraint(fleet),
                rule.build_expenditure_constraint(benchmark_expenditure),
            ),
        )

        if best_insertion is None and use_pickup_filter:
            return _insert_request_lowest_cost(
                request,
                allowed_vehicle_ids,
                use_pickup_filter=False,
            )

        if best_insertion is None:
            return False

        vehicle_id = int(best_insertion["vehicle_id"])
        next_trip_id = drt.start_or_update_trip(
            vehicle_states,
            vehicle_id,
            dict(best_insertion["evaluation"]),
            list(best_insertion["schedule"]),
            next_trip_id,
        )
        vehicle_states[vehicle_id].pending_events = list(best_insertion["schedule"])
        vehicle_states[vehicle_id].pending_evaluation = dict(best_insertion["evaluation"])
        vehicle_states[vehicle_id].has_departed = True
        scheduled_request_ids.add(request.request_id)
        return True

    if prebooking_requests:
        pre_clusters,noise = _dbscan_clusters(
            prebooking_requests,
            network_context.graph,
        )
    else:
        pre_clusters = []
        noise = []

    for cluster_index, cluster in enumerate(pre_clusters):
        vehicle_id = prebooking_vehicle_ids[cluster_index % len(prebooking_vehicle_ids)] # 将预订请求分配给预订车辆，基于聚类结果和预订车辆数量，使用modulo操作循环分配
        for request in sorted(cluster, key=lambda trip: trip.departure_time):
            if not _insert_request_lowest_cost(request, (vehicle_id,)):
                # raise ValueError('prebooking isnt served')
                pass

    if noise != []:
        for request in noise:
            if _insert_request_lowest_cost(request, prebooking_vehicle_ids):
                continue
            if _insert_request_lowest_cost(request, all_vehicle_ids):
                continue
            skipped_request_ids.add(request.request_id)

    for request in sorted(realtime_requests, key=lambda trip: trip.departure_time):
        drt.advance_vehicle_states(
            vehicle_states,
            int(request.departure_time),
            network_context.graph,
            nets,
            fleet,
        )
        if _insert_request_lowest_cost(request, reserved_vehicle_ids):
            continue
        if _insert_request_lowest_cost(request, all_vehicle_ids):
            continue
        skipped_request_ids.add(request.request_id)

    sync_accumulator()
    return rule.finalize_nonbaseline_mode(
        mode_id=3,
        scenario=scenario,
        requests=requests,
        benchmark_expenditure=benchmark_expenditure,
        acc=acc,
        feasible=True,
        feasibility_reason=rule.request_type_mode_reason(acc.served_requests, len(requests)),
    )


def evaluate_4( # 评估枢纽辐射模式 hub-and-spoke
    config, nets, fleet,
    requests: list[TripRequest],
    scenario: Scenario,
    network_context,
    benchmark_expenditure: float | None,
    service_policy: str = "strict",
) -> dict[str, Any]:

    service_policy = rule.validate_service_policy(service_policy)
    acc = rule.init_mode_accumulator()
    prebooking_requests, realtime_requests = requests_by_type(requests)

    loops, weighted_contexts, fixed_metrics = fix.build_context(network_context, config, nets, fleet)
    if not fixed_metrics["feasible"]:
        return rule.finalize_nonbaseline_mode(
            mode_id=4,
            scenario=scenario,
            requests=requests,
            benchmark_expenditure=benchmark_expenditure,
            acc=acc,
            feasible=False,
            feasibility_reason=str(fixed_metrics["feasibility_reason"]),
        )

    loads: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    rule.set_operator_metrics(
        acc,
        float(fixed_metrics["total_travel_distance"]),
        int(fixed_metrics["total_trips"]),
    )

    def insert_request(request: TripRequest) -> bool:
        loop = fixed_loop._nearest_loop_for_request(request, loops, network_context.graph)
        loop_context = weighted_contexts[loop.id]
        stop_offsets = loop_context["offsets"]
        route_length = float(loop_context["length"])
        candidates: list[dict[str, Any]] = []

        for boarding_stop, boarding_index in loop.fixed_stop_indices.items():
            origin_walk_d = fixed_loop.distance_shortpath(request.origin, boarding_stop, network_context.graph)
            boarding_offset = float(stop_offsets[boarding_stop])
            for alighting_stop, alighting_index in loop.fixed_stop_indices.items():
                destination_walk_d = fixed_loop.distance_shortpath(
                    request.destination,
                    alighting_stop,
                    network_context.graph,
                )
                walk_time = float(origin_walk_d + destination_walk_d)
                alighting_offset = float(stop_offsets[alighting_stop])
                onboard_time = float(
                    fixed_loop._calculate_travel_weighted(
                        boarding_offset,
                        alighting_offset,
                        route_length,
                    )
                )

                for scheduled_pass in fixed_loop._scheduled_pass_candidates(
                    config,
                    fleet,
                    loop,
                    boarding_offset,
                    request.departure_time,
                ):
                    vehicle_id = int(scheduled_pass["vehicle_id"])
                    boarding_time = float(scheduled_pass["pass_time"])
                    wait_time = float(boarding_time - request.departure_time)
                    if not fixed_loop._check_loop_capacity(
                        fleet,
                        loads,
                        vehicle_id,
                        boarding_time,
                        boarding_index,
                        alighting_index,
                        loop.length,
                    ):
                        continue

                    candidate_trips = int(fixed_metrics["total_trips"])
                    candidate_travel_distance = float(
                        fixed_metrics["total_travel_distance"]
                    )
                    candidate_expenditure = rule._calculate_net_expenditure(
                        candidate_travel_distance,
                        candidate_departures,
                        acc.served_requests + 1,
                    )
                    ranking = (
                        wait_time + walk_time + onboard_time,
                        wait_time,
                        walk_time,
                        onboard_time,
                        loop.id,
                        vehicle_id,
                    )
                    candidates.append(
                        {
                            "vehicle_id": vehicle_id,
                            "boarding_index": boarding_index,
                            "alighting_index": alighting_index,
                            "boarding_time": boarding_time,
                            "wait_time": wait_time,
                            "walk_time": walk_time,
                            "onboard_time": onboard_time,
                            "candidate_departures": candidate_departures,
                            "candidate_travel_distance": candidate_travel_distance,
                            "candidate_expenditure": candidate_expenditure,
                            "route_length": loop.length,
                            "objective": (
                                acc.total_wait
                                + acc.total_walk
                                + acc.total_onboard
                                + wait_time
                                + walk_time
                                + onboard_time
                            ),
                            "ranking": ranking,
                        }
                    )

        best_choice, _ = rule.minimize_objective(candidates, ())
        if best_choice is None:
            return False

        fixed_loop._reserve_loop_capacity(
            loads,
            int(best_choice["vehicle_id"]),
            float(best_choice["boarding_time"]),
            int(best_choice["boarding_index"]),
            int(best_choice["alighting_index"]),
            int(best_choice["route_length"]),
        )

        acc.served_requests += 1
        acc.total_wait += float(best_choice["wait_time"])
        acc.total_walk += float(best_choice["walk_time"])
        acc.total_onboard += float(best_choice["onboard_time"])
        rule.set_operator_metrics(
            acc,
            float(best_choice["candidate_travel_distance"]),
            int(best_choice["candidate_departures"]),
        )
        return True

    for request in prebooking_requests:
        if not insert_request(request):
            return rule.finalize_nonbaseline_mode(
                mode_id=4,
                scenario=scenario,
                requests=requests,
                benchmark_expenditure=benchmark_expenditure,
                acc=acc,
                feasible=False,
                feasibility_reason="prebooking_insertion_failed",
            )

    for request in realtime_requests:
        insert_request(request)

    return rule.finalize_nonbaseline_mode(
        mode_id=4,
        scenario=scenario,
        requests=requests,
        benchmark_expenditure=benchmark_expenditure,
        acc=acc,
        feasible=True,
        feasibility_reason=rule.request_type_mode_reason(acc.served_requests, len(requests)),
    )

# 使用DBSCAN算法对请求进行聚类，
def _dbscan_clusters(
    requests: list[TripRequest],
    graph: nx.Graph,
) -> list[list[TripRequest]]: 
    # 基于请求间的departure_time、origin网络距离、destination网络距离进行聚类

    ordered_requests = sorted(requests, key=lambda request: request.departure_time) 
    distance_matrix = _dbscan_matrix(ordered_requests, graph) # 计算请求之间的距离矩阵，结合时间距离和空间距离
    labels = DBSCAN(
        eps=0.8,
        min_samples=2,
        metric="precomputed",
    ).fit_predict(distance_matrix)

    clusters_by_label: dict[int, list[TripRequest]] = defaultdict(list)
    noise_cluster: list[TripRequest] = []
    for request, label in zip(ordered_requests, labels):
        label = int(label)
        if label == -1:
            noise_cluster.append(request)
        else:
            clusters_by_label[label].append(request)

    normal_clusters = [cluster for cluster in clusters_by_label.values() if cluster]
    sorted_clusters = sorted(
        normal_clusters,
        key=lambda cluster: cluster[0].departure_time,
    )
    if noise_cluster:
        return [sorted_clusters, noise_cluster]
    return sorted_clusters, []

def _dbscan_matrix(ordered_requests,
                   graph,
        
):
    request_count = len(ordered_requests)
    time_distances = np.zeros((request_count, request_count), dtype=float)
    origin_distances = np.zeros((request_count, request_count), dtype=float)
    destination_distances = np.zeros((request_count, request_count), dtype=float)

    for i in range(request_count):
        request_i = ordered_requests[i]
        for j in range(i + 1, request_count):
            request_j = ordered_requests[j]
            time_distance = float(
                abs(request_i.departure_time - request_j.departure_time)
            )
            origin_distance = fixed_loop.distance_shortpath(
                request_i.origin,
                request_j.origin,
                graph,
            )
            destination_distance = fixed_loop.distance_shortpath(
                request_i.destination,
                request_j.destination,
                graph,
            )
            time_distances[i, j] = time_distance
            time_distances[j, i] = time_distance
            origin_distances[i, j] = origin_distance
            origin_distances[j, i] = origin_distance
            destination_distances[i, j] = destination_distance
            destination_distances[j, i] = destination_distance

    time_scale = float(time_distances.max()) or 1.0
    origin_scale = float(origin_distances.max()) or 1.0
    destination_scale = float(destination_distances.max()) or 1.0
    distance_matrix = (
        5*time_distances / time_scale
        + origin_distances / origin_scale
        + destination_distances / destination_scale
    )
    return distance_matrix


def _sorted_requests(requests: list[TripRequest]) -> list[TripRequest]:
    return sorted(requests, key=lambda request: (request.departure_time, request.request_id))


def requests_by_type(
    requests: list[TripRequest],
) -> tuple[list[TripRequest], list[TripRequest]]: # 将请求分为预订请求和实时请求两类，基于request_type属性进行划分，并保持每类内的请求按照departure_time排序
    prebooking_requests: list[TripRequest] = []
    realtime_requests: list[TripRequest] = []
    for request in _sorted_requests(requests):
        if request.request_type == 1:
            prebooking_requests.append(request)
        else:
            realtime_requests.append(request)
    return prebooking_requests, realtime_requests

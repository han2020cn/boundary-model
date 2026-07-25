from __future__ import annotations

from collections import defaultdict
from typing import Any
import helpers.common_rule as rule
import helpers.deviation as deviation
import helpers.drt as drt 
import helpers.fixed_loop as fixed_loop

from helpers.config import TripRequest
from helpers.types import Scenario


def _travel_time(distance: float, fleet) -> float:
    return float(distance) / float(fleet.speed)


def build_baseline(
    config,
    nets,
    fleet,
    network_context,
    scenario: Scenario,
    requests: list[TripRequest],
    loop_context,

) -> dict[str, Any]:
    walk_v = config.walk_speed
    bus_v = fleet.speed
    loops, weighted_contexts, fixed_metrics = loop_context
    if not fixed_metrics["feasible"]:
        raise ValueError("baseline is not feasible")

    loads: defaultdict[fixed_loop.LoopLoadKey, int] = defaultdict(int)
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
        onboard_t = float(base_route_d) / float(bus_v)
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
            departure_index = int(scheduled_pass["departure_index"])
            trip_key: fixed_loop.PhysicalTripKey = (
                str(loop.id),
                vehicle_id,
                departure_index,
            )
            boarding_time = float(scheduled_pass["pass_time"])
            wait_time = float(boarding_time - earliest_boarding_time)

            if not fixed_loop._check_loop_capacity(
                fleet,
                loads,
                trip_key,
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
                "departure_index": departure_index,
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
                "next_assigned_departure_time": fixed_loop.next_assigned_departure_time(
                    config,
                    fleet,
                    loop,
                    int(scheduled_pass["departure_index"]),
                ),
            }
            break

        if best_choice is None:
            # raise ValueError(f"no feasible pass found for request {request.request_id} on loop {loop.id}")
            continue

        fixed_loop._reserve_loop_capacity(
            loads,
            (
                str(best_choice["loop_id"]),
                int(best_choice["vehicle_id"]),
                int(best_choice["departure_index"]),
            ),
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
        float(fixed_metrics["operating_time"]),
    )
    #baseline排除了departtime 无法服务的request，所以total_requests = served_requests
    result = rule._finalize_result(
        mode_id=1,
        scenario=scenario,
        total_requests = served_requests,
        served_requests = served_requests,
        benchmark_expenditure=net_expenditure,
        net_expenditure=net_expenditure,
        total_wait=all_wait,
        total_walk=all_walk,
        total_onboard=all_onboard,
        feasible=True,
        feasibility_reason="feasible",
        total_trips=int(fixed_metrics["total_trips"]),
        operating_time=float(fixed_metrics["operating_time"]),
    )
    result["benchmark_expenditure"] = result["net_expenditure"]
    return {
        "result": result,
        "assignments": assignments,
        "loops": loops,
        "weighted_contexts": weighted_contexts,
        "fixed_metrics": fixed_metrics,
    }

# 评估偏离路线模式 deviated route
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
    return deviation.deviation_2(
        config,
        nets,
        fleet,
        baseline,
        requests,
        scenario,
        network_context,
        benchmark_expenditure,
        service_policy,
    )


def evaluate_3(
    config,
    nets,
    fleet,
    requests: list[TripRequest],
    scenario: Scenario,
    network_context,
    benchmark_expenditure: float | None,
    service_policy: str = "strict",
) -> dict[str, Any]:
    del nets
    service_policy = rule.validate_service_policy(service_policy)
    acc = rule.init_mode_accumulator()

    if fleet.num <= 0:
        return rule.finalize_nonbaseline_mode(
            mode_id=3,
            scenario=scenario,
            requests=requests,
            benchmark_expenditure=benchmark_expenditure,
            acc=acc,
            feasible=False,
            feasibility_reason="no_vehicle",
        )
    if benchmark_expenditure is None:
        raise ValueError("benchmark_expenditure must be provided")


    try:
        policy = drt.build_policy(config, benchmark_expenditure)
        context = drt.build_context(
            network_context=network_context,
            fleet=fleet,
            policy=policy,
            expenditure_fn=rule._calculate_net_expenditure,
        )
    except (TypeError, ValueError) as exc:
        return rule.finalize_nonbaseline_mode(
            mode_id=3,
            scenario=scenario,
            requests=requests,
            benchmark_expenditure=benchmark_expenditure,
            acc=acc,
            feasible=False,
            feasibility_reason=str(exc),
        )

    state = drt.initialize_system(context)
    prebooking_requests, realtime_requests = requests_by_type(requests)
    clusters = drt.cluster_prebookings(prebooking_requests, context)
    drt.plan_prebookings(
        state,
        clusters,
        context,
        service_policy=service_policy,
    )

    rejected_prebookings = {
        request.request_id
        for request in prebooking_requests
        if request.request_id in state.rejected_requests
    }
    if service_policy == "strict" and rejected_prebookings:
        summary = drt.summarize_system(state, context)
        acc.served_requests = summary.served_requests
        acc.total_wait = summary.total_wait
        acc.total_walk = 0.0
        acc.total_onboard = summary.total_onboard
        rule.set_operator_metrics(
            acc,
            summary.total_travel_distance,
            summary.total_trips,
            summary.operating_time,
        )
        acc.max_concurrent_trips = summary.max_concurrent_trips
        acc.vehicle_reuse_ratio = summary.vehicle_reuse_ratio
        return rule.finalize_nonbaseline_mode(
            mode_id=3,
            scenario=scenario,
            requests=requests,
            benchmark_expenditure=benchmark_expenditure,
            acc=acc,
            feasible=False,
            feasibility_reason="prebooking_infeasible",
        )

    for epoch, batch in drt.build_realtime_batches(realtime_requests, context):
        drt.advance_system_to_epoch(state, epoch, context)
        for request in batch:
            drt.insert_realtime_request(
                state,
                request,
                epoch,
                context,
            )

    drt.advance_system_to_epoch(
        state,
        context.policy.latest_return_time,
        context,
    )
    system_valid, system_violations = drt.validate_system(state, context)
    summary = drt.summarize_system(state, context)
    acc.served_requests = summary.served_requests
    acc.total_wait = summary.total_wait
    acc.total_walk = 0.0
    acc.total_onboard = summary.total_onboard
    rule.set_operator_metrics(
        acc,
        summary.total_travel_distance,
        summary.total_trips,
        summary.operating_time,
    )
    acc.max_concurrent_trips = summary.max_concurrent_trips
    acc.vehicle_reuse_ratio = summary.vehicle_reuse_ratio

    all_requests_served = summary.served_requests == len(requests)
    within_budget = (
        summary.net_expenditure
        <= benchmark_expenditure + policy.tolerance
    )
    feasible = (
        system_valid
        and within_budget
        and (service_policy != "strict" or all_requests_served)
    )
    if not system_valid:
        feasibility_reason = (
            system_violations[0]
            if system_violations
            else "drt_system_invalid"
        )
    elif service_policy == "strict" and not all_requests_served:
        feasibility_reason = "unserved_requests"
    else:
        feasibility_reason = rule.request_type_mode_reason(
            summary.served_requests,
            len(requests),
        )
    return rule.finalize_nonbaseline_mode(
        mode_id=3,
        scenario=scenario,
        requests=requests,
        benchmark_expenditure=benchmark_expenditure,
        acc=acc,
        feasible=feasible,
        feasibility_reason=feasibility_reason,
    )

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

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable
import math
import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
import drt 
import functions as fs

from config import LoopContext, SpokeVehicle,  ModeAccumulator, TripRequest


REFUSAL_PENALTY = 0.0
#type alias（类型别名），作用是给类型标注起更有业务含义的名字
DISTANCE_CACHE_KEY = "_boundary_model_shortest_path_lengths"
NetworkNode = Any # network node（网络节点）
GridNode = NetworkNode
Scenario = dict[str, Any]
CandidateConstraint = Callable[[dict[str, Any]], str | None]


#module-level global variables（模块级全局变量）
#通过函数被赋值 mode_set.configure_runtime(config, nets, fleet, network_context)
config: Any | None = None
nets: Any | None = None
fleet: Any | None = None
network_context: Any | None = None

#TODO
# 计算净支出 constraint
def _calculate_net_expenditure( 
    total_travel_distance: float,
    total_departures: int,
    served_requests: int,
) -> float:
    return float(total_travel_distance * 3.17 + total_departures * 1000 - served_requests*3) # 单位£ 运营商的总支出 = 车辆行驶的总距离 * 每英里成本 + 车辆出发的总次数 * 每次出发成本 - 服务的请求数量 * 每个请求的收益

# minimize objective
def _calculate_objective(
    total_wait: float,
    total_walk: float,
    total_onboard: float,
    total_requests: int,
    served_requests: int,
) -> float:
    refusal_requests = int(total_requests - served_requests)
    return float(2*total_wait + 3*total_walk + total_onboard + REFUSAL_PENALTY * refusal_requests)

def _max_deviation_mode2() -> float: # 计算模式2的最大偏离比例，基于nets.grid_len和config.max_dev的值，确保偏离不超过网格长度的一半
    grid_len = float(getattr(nets, "grid_len"))
    if grid_len <= 0.0:
        raise ValueError("nets.grid_len must be positive for deviated route mode")

    configured_max_dev = float(getattr(config, "max_dev", 0.5 * grid_len))
    if configured_max_dev < 0.0:
        raise ValueError("config.max_dev must be non-negative for deviated route mode")

    effective_max_dev = min(configured_max_dev, 0.5 * grid_len)
    return effective_max_dev / grid_len


def _minimize_objective( # 在候选方案中选择满足约束条件且objective最小的一个，返回该方案和未满足约束的原因集合
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

#TODO
def configure_runtime(
    runtime_config: Any,
    runtime_nets: Any,
    runtime_fleet: Any,
    runtime_network_context: Any | None = None,
) -> None:
    global config, nets, fleet, network_context
    config = runtime_config
    nets = runtime_nets
    fleet = runtime_fleet
    network_context = runtime_network_context


def _require_runtime() -> None:
    if config is None or nets is None or fleet is None:
        raise RuntimeError("mode_set runtime classes must be configured before evaluation")




def distance_shortpath( # 计算两个节点之间的最短路径距离，基于 graph（图）中的 edge weight（边权重）
    a: GridNode,
    b: GridNode,
    net_graph: nx.Graph,
) -> float:
    distance_cache = net_graph.graph.get(DISTANCE_CACHE_KEY)
    if distance_cache is None:
        distance_cache = {
            source: dict(lengths)
            for source, lengths in nx.all_pairs_dijkstra_path_length(
                net_graph,
                weight="weight",
            )
        }
        net_graph.graph[DISTANCE_CACHE_KEY] = distance_cache

    return float(distance_cache[a][b])



def _init_mode_accumulator() -> ModeAccumulator:
    return ModeAccumulator()



def _set_operator_metrics(    #更新运营商相关的累计指标
    acc: ModeAccumulator,
    total_travel_distance: float,
    total_departures: int,
) -> None:
    acc.total_travel_distance = float(total_travel_distance)
    acc.total_departures = int(total_departures)
    acc.net_expenditure = _calculate_net_expenditure(
        acc.total_travel_distance,
        acc.total_departures,
        acc.served_requests,
    )


def _loop_departure_count(completion_time: float, cycle_length: int) -> int: # 计算循环路线的出发次数，基于完成时间和循环周期长度，向上取整得到出发次数
    if completion_time <= 0.0:
        return 0
    return int(math.ceil(completion_time / cycle_length))

#TODO 是否可以简化
def _finalize_nonbaseline_mode( # 将非基线模式的评估结果整理成一个字典，包含各种指标和可行性信息
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
        max_concurrent_trips=acc.max_concurrent_trips,
        vehicle_reuse_ratio=acc.vehicle_reuse_ratio,
    )



def _validate_service_policy(service_policy: str) -> str:
    if service_policy not in {"strict", "skip"}:
        raise ValueError("service_policy must be 'strict' or 'skip'")
    return service_policy





def _build_loop_capacity_constraint(  # 构建一个约束函数，用于检查候选方案是否满足循环路线的容量限制
    runtime_fleet: Any,
    loads: defaultdict[tuple[int, int, float], int],
    route_length: int,
) -> CandidateConstraint:
    def constraint(candidate: dict[str, Any]) -> str | None:
        if _check_loop_capacity(
            runtime_fleet,
            loads,
            int(candidate["vehicle_id"]),
            float(candidate["route_start_time"]),
            int(candidate["boarding_anchor"]),
            int(candidate["alighting_anchor"]),
            route_length,
        ):
            return None
        return "capacity_limit"

    return constraint





def _build_expenditure_constraint( # 构建一个约束函数，用于检查候选方案的预期支出是否超过基准支出
    benchmark_expenditure: float | None,
) -> CandidateConstraint:
    def constraint(candidate: dict[str, Any]) -> str | None:
        if benchmark_expenditure is None:
            return None
        if float(candidate["candidate_expenditure"]) <= benchmark_expenditure:
            return None
        return "benchmark_exceeded"

    return constraint


def _evaluate_1( # 评估固定路线模式
    requests: list[TripRequest],
    scenario: Scenario,
    graph: nx.Graph,
) -> dict[str, Any]:
    _require_runtime()
    loops = _build_loop_all(graph)
    loads: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    vehicle_completion = {vehicle_id: 0.0 for vehicle_id in range(fleet.num)}
    served_requests = 0
    total_wait = 0.0
    total_walk = 0.0
    total_onboard = 0.0
    #对每个request循环
    for request in _sorted_requests(requests):
        loop = _nearest_loop_for_request(request, loops, graph)
        candidates: list[dict[str, Any]] = []

        for boarding_stop, boarding_index in loop.fixed_stop_indices.items():
            origin_walk = distance_shortpath(request.origin, boarding_stop, graph)
            for alighting_stop, alighting_index in loop.fixed_stop_indices.items():
                destination_walk = distance_shortpath(request.destination, alighting_stop, graph)
                walk_time = float(origin_walk + destination_walk)
                onboard_time = float(
                    _circular_travel_time(
                        boarding_index,
                        alighting_index,
                        loop.length,
                    )
                )

                for vehicle_id, offset in loop.vehicle_offsets.items():
                    first_pass = (boarding_index - offset) % loop.length
                    boarding_time = _next_cyclic_pass(
                        request.departure_time,
                        first_pass,
                        loop.length,
                    )
                    wait_time = float(boarding_time - request.departure_time)
                    if not _check_loop_capacity(
                        fleet,
                        loads,
                        vehicle_id,
                        boarding_time,
                        boarding_index,
                        alighting_index,
                        loop.length,
                    ):
                        continue

                    operator_finish = float(boarding_time + onboard_time)
                    # 排序选择best Choice
                    ranking = (
                        wait_time + walk_time + onboard_time,
                        wait_time,
                        walk_time,
                        onboard_time,
                    )
                    candidates.append(
                        {
                            "vehicle_id": vehicle_id,
                            "boarding_index": boarding_index,
                            "alighting_index": alighting_index,
                            "boarding_time": boarding_time,
                            "operator_finish": operator_finish,
                            "wait_time": wait_time,
                            "walk_time": walk_time,
                            "onboard_time": onboard_time,
                            "objective": (
                                total_wait
                                + total_walk
                                + total_onboard
                                + wait_time
                                + walk_time
                                + onboard_time
                            ),
                            "ranking": ranking,
                        }
                    )

        best_choice, _ = _minimize_objective(candidates, ())

        if best_choice is None:
            total_travel_distance = float(sum(vehicle_completion.values()))
            total_departures = _loop_departures_by_vehicle(vehicle_completion, loops)
            result = _finalize_result(
                mode_id=1,
                scenario=scenario,
                total_requests=len(requests),
                served_requests=served_requests,
                benchmark_expenditure=None,
                net_expenditure=_calculate_net_expenditure(
                    total_travel_distance,
                    total_departures,
                    served_requests,
                ),
                total_wait=total_wait,
                total_walk=total_walk,
                total_onboard=total_onboard,
                feasible=False,
                feasibility_reason="capacity_limit",
            )
            result["benchmark_expenditure"] = None
            return result

        _reserve_loop_capacity(
            loads,
            best_choice["vehicle_id"],
            best_choice["boarding_time"],
            best_choice["boarding_index"],
            best_choice["alighting_index"],
            loop.length,
        )
        vehicle_id = int(best_choice["vehicle_id"])
        vehicle_completion[vehicle_id] = max(
            vehicle_completion[vehicle_id],
            float(best_choice["operator_finish"]),
        )
        served_requests += 1
        total_wait += float(best_choice["wait_time"])
        total_walk += float(best_choice["walk_time"])
        total_onboard += float(best_choice["onboard_time"])

    total_travel_distance = float(sum(vehicle_completion.values()))
    total_departures = _loop_departures_by_vehicle(vehicle_completion, loops)
    net_expenditure = _calculate_net_expenditure(
        total_travel_distance,
        total_departures,
        served_requests,
    )
    result = _finalize_result(
        mode_id=1,
        scenario=scenario,
        total_requests=len(requests),
        served_requests=served_requests,
        benchmark_expenditure=net_expenditure,
        net_expenditure=net_expenditure,
        total_wait=total_wait,
        total_walk=total_walk,
        total_onboard=total_onboard,
        feasible=True,
        feasibility_reason="feasible",
    )
    result["benchmark_expenditure"] = result["net_expenditure"]
    return result


def _evaluate_2( # 评估偏离路线模式 deviated route
    requests: list[TripRequest],
    scenario: Scenario,
    graph: nx.Graph,
    benchmark_expenditure: float | None,
    service_policy: str = "strict",
) -> dict[str, Any]:

    _require_runtime()
    service_policy = _validate_service_policy(service_policy)
    acc = _init_mode_accumulator() # 初始化一个ModeAccumulator对象acc，用于累计评估指标
    prebooking_requests, realtime_requests = requests_by_type(requests)  #将请求按照类型分为预订请求和实时请求
    loops = _build_loop_all(graph)
    loads: defaultdict[tuple[int, int, float], int] = defaultdict(int)  #
    vehicle_completion = {vehicle_id: 0.0 for vehicle_id in range(fleet.num)}
    vehicle_delay = {vehicle_id: 0.0 for vehicle_id in range(fleet.num)}

    def insert_request(request: TripRequest) -> bool:
        candidates: list[dict[str, Any]] = []
        loop = _nearest_loop_for_request(request, loops, graph)
        boarding_locations = _deviation_locs(request.origin, loop, graph)
        alighting_locations = _deviation_locs(request.destination, loop, graph)

        for boarding_location in boarding_locations:
            boarding_anchor = int(boarding_location["anchor_index"])
            boarding_deviation = float(boarding_location["vehicle_deviation"])
            boarding_extra_walk = float(boarding_location["extra_walk"])

            for alighting_location in alighting_locations:
                alighting_anchor = int(alighting_location["anchor_index"])
                alighting_deviation = float(alighting_location["vehicle_deviation"])
                alighting_extra_walk = float(alighting_location["extra_walk"])
                walk_time = float(boarding_extra_walk + alighting_extra_walk)
                base_route_time = _circular_travel_time(
                    boarding_anchor,
                    alighting_anchor,
                    loop.length,
                )
                passenger_onboard = float(
                    base_route_time
                    + boarding_deviation
                    + alighting_deviation
                )
                passenger_board_offset = boarding_deviation
                route_entry_offset = 2.0 * boarding_deviation
                operator_finish_offset = (
                    route_entry_offset
                    + base_route_time
                    + 2.0 * alighting_deviation
                )
                added_delay = route_entry_offset + (2.0 * alighting_deviation)

                for vehicle_id, offset in loop.vehicle_offsets.items():
                    delayed_first_pass = (
                        (boarding_anchor - offset) % loop.length
                    ) + vehicle_delay[vehicle_id]
                    anchor_time = _next_cyclic_pass(
                        request.departure_time,
                        delayed_first_pass,
                        loop.length,
                    )
                    passenger_board_time = anchor_time + passenger_board_offset
                    route_start_time = anchor_time + route_entry_offset
                    wait_time = float(passenger_board_time - request.departure_time)

                    candidate_completion = max(
                        vehicle_completion[vehicle_id],
                        float(anchor_time + operator_finish_offset),
                    )
                    candidate_travel_distance = (
                        acc.total_travel_distance
                        - vehicle_completion[vehicle_id]
                        + candidate_completion
                    )
                    candidate_departures = (
                        acc.total_departures
                        - _loop_departure_count(vehicle_completion[vehicle_id], loop.length)
                        + _loop_departure_count(candidate_completion, loop.length)
                    )
                    candidate_expenditure = _calculate_net_expenditure( #计算净支出
                        candidate_travel_distance,
                        candidate_departures,
                        acc.served_requests + 1,
                    )

                    ranking = (
                        wait_time + walk_time + passenger_onboard,
                        wait_time,
                        walk_time,
                        passenger_onboard,
                        boarding_deviation + alighting_deviation,
                        boarding_extra_walk + alighting_extra_walk,
                    )
                    candidates.append(
                        {
                            "vehicle_id": vehicle_id,
                            "boarding_anchor": boarding_anchor,
                            "alighting_anchor": alighting_anchor,
                            "route_start_time": route_start_time,
                            "candidate_completion": candidate_completion,
                            "candidate_travel_distance": candidate_travel_distance,
                            "candidate_departures": candidate_departures,
                            "candidate_expenditure": candidate_expenditure,
                            "added_delay": added_delay,
                            "wait_time": wait_time,
                            "walk_time": walk_time,
                            "onboard_time": passenger_onboard,
                            "objective": (
                                acc.total_wait
                                + acc.total_walk
                                + acc.total_onboard
                                + wait_time
                                + walk_time
                                + passenger_onboard
                            ),
                            "ranking": ranking,
                        }
                    )

        best_choice, _ = _minimize_objective(
            candidates,
            (
                _build_loop_capacity_constraint(fleet, loads, loop.length),
                _build_expenditure_constraint(benchmark_expenditure),
            ),
        )

        if best_choice is None: 
            return False

        _reserve_loop_capacity(
            loads,
            int(best_choice["vehicle_id"]),
            float(best_choice["route_start_time"]),
            int(best_choice["boarding_anchor"]),
            int(best_choice["alighting_anchor"]),
            loop.length,
        )
        vehicle_id = int(best_choice["vehicle_id"])
        vehicle_completion[vehicle_id] = float(best_choice["candidate_completion"])
        vehicle_delay[vehicle_id] += float(best_choice["added_delay"])
        acc.served_requests += 1
        acc.total_wait += float(best_choice["wait_time"])
        acc.total_walk += float(best_choice["walk_time"])
        acc.total_onboard += float(best_choice["onboard_time"])
        _set_operator_metrics( # 更新运营商相关的累计指标
            acc,
            float(best_choice["candidate_travel_distance"]),
            int(best_choice["candidate_departures"]),
        )
        return True

    for request in prebooking_requests:
        if not insert_request(request):
            return _finalize_nonbaseline_mode(
                mode_id=2,
                scenario=scenario,
                requests=requests,
                benchmark_expenditure=benchmark_expenditure,
                acc=acc,
                feasible=False,
                feasibility_reason="prebooking_insertion_failed",
            )

    for request in realtime_requests:
        insert_request(request)

    return _finalize_nonbaseline_mode(
        mode_id=2,
        scenario=scenario,
        requests=requests,
        benchmark_expenditure=benchmark_expenditure,
        acc=acc,
        feasible=True,
        feasibility_reason=_request_type_mode_reason(acc.served_requests, len(requests)),
    )






def _evaluate_3( # 评估动态路线模式 DRT with clustered pre-booking（聚类预订）
    requests: list[TripRequest],
    scenario: Scenario,
    graph: nx.Graph,
    benchmark_expenditure: float | None,
    service_policy: str = "strict",
) -> dict[str, Any]:
    _require_runtime()
    service_policy = _validate_service_policy(service_policy)
    acc = _init_mode_accumulator()
    prebooking_requests, realtime_requests = requests_by_type(requests)
    vehicle_states: dict[int, drt.DrtVehicleState] = {
        vehicle_id: drt.DrtVehicleState(current_location=_runtime_hub())
        for vehicle_id in range(fleet.num)
    }

    for state in vehicle_states.values():
        state.pending_evaluation = drt._evaluate_drt_event_schedule(
            [],
            graph,
            nets,
            fleet,
            start_location=_runtime_hub(),
        )

    scheduled_request_ids: set[int] = set()
    skipped_request_ids: set[int] = set()
    all_vehicle_ids = tuple(vehicle_states)
    next_trip_id = 1
    if not all_vehicle_ids:
        return _finalize_nonbaseline_mode(
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
    travel_weight = float(getattr(config, "drt_travel_weight", 1.0))

    def sync_accumulator() -> None: # 同步accumulator的指标，基于当前车辆状态和已安排的请求
        totals = drt._drt_state_totals(vehicle_states)
        trips = drt._drt_trips_by_state(vehicle_states)
        acc.served_requests = len(scheduled_request_ids)
        acc.total_wait = totals["wait"]
        acc.total_walk = 0.0
        acc.total_onboard = totals["onboard"]
        _set_operator_metrics(
            acc,
            totals["travel_distance"],
            int(totals["departures"]),
        )
        acc.total_trips = int(totals["departures"])
        acc.max_concurrent_trips = drt._max_concurrent_trips(trips)
        acc.vehicle_reuse_ratio = (
            None
            if fleet.num <= 0
            else float(acc.total_trips) / float(fleet.num)
        )

    def advance_vehicles_to(planning_time: int) -> None:
        for state in vehicle_states.values():
            drt._advance_drt_vehicle_state(
                state,
                planning_time,
                graph,
                nets,
                fleet,
            )

    def start_or_update_trip(
        vehicle_id: int,
        evaluation: dict[str, Any],
        schedule: list[drt.DrtEvent],
    ) -> None:
        nonlocal next_trip_id
        state = vehicle_states[vehicle_id]
        request_ids = drt._event_request_ids(schedule)
        if drt._state_has_active_trip(state):
            drt._add_drt_trip_request_ids(state, request_ids)
            return
        drt._start_drt_trip(
            state,
            trip_id=next_trip_id,
            vehicle_id=vehicle_id,
            start_time=drt._trip_start_time_from_evaluation(
                evaluation,
                state.current_time,
            ),
            start_location=state.current_location,
            request_ids=request_ids,
        )
        next_trip_id += 1
    #TODO
    def _insert_request_lowest_cost( # 尝试将请求插入到允许的车辆中，选择增加的travel distance（行驶距离）最小的插入方式，如果成功插入返回True，否则返回False
        request: TripRequest,
        allowed_vehicle_ids: tuple[int, ...],
        *,
        use_pickup_filter: bool = True,
    ) -> bool:
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
                        graph,
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
                    graph,
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
                    current_totals["departures"]
                    + int(not drt._state_has_active_trip(state))
                )
                candidate_expenditure = _calculate_net_expenditure(
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
                    + travel_weight * delta_travel
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

        best_insertion, _ = _minimize_objective(
            candidates,
            (
                drt._build_drt_capacity_constraint(fleet),
                _build_expenditure_constraint(benchmark_expenditure),
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
        start_or_update_trip(
            vehicle_id,
            dict(best_insertion["evaluation"]),
            list(best_insertion["schedule"]),
        )
        vehicle_states[vehicle_id].pending_events = list(best_insertion["schedule"])
        vehicle_states[vehicle_id].pending_evaluation = dict(best_insertion["evaluation"])
        vehicle_states[vehicle_id].has_departed = True
        scheduled_request_ids.add(request.request_id)
        return True

    if prebooking_requests:
        pre_clusters,noise = _dbscan_clusters(
            prebooking_requests,
            graph,
        )
    else:
        pre_clusters = []
        noise = []

    for cluster_index, cluster in enumerate(pre_clusters):
        vehicle_id = prebooking_vehicle_ids[cluster_index % len(prebooking_vehicle_ids)] # 将预订请求分配给预订车辆，基于聚类结果和预订车辆数量，使用modulo操作循环分配
        for request in sorted(cluster, key=lambda item: _request_sort_key(item, graph)):
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
            
    for request in sorted(realtime_requests, key=lambda item: _request_sort_key(item, graph)):
        advance_vehicles_to(int(request.departure_time))
        if _insert_request_lowest_cost(request, reserved_vehicle_ids):
            continue
        if _insert_request_lowest_cost(request, all_vehicle_ids):
            continue
        skipped_request_ids.add(request.request_id)

    sync_accumulator()
    return _finalize_nonbaseline_mode(
        mode_id=3,
        scenario=scenario,
        requests=requests,
        benchmark_expenditure=benchmark_expenditure,
        acc=acc,
        feasible=True,
        feasibility_reason=_request_type_mode_reason(acc.served_requests, len(requests)),
    )


def _evaluate_4( # 评估枢纽辐射模式 hub-and-spoke
    requests: list[TripRequest],
    scenario: Scenario,
    graph: nx.Graph,
    benchmark_expenditure: float | None,
    service_policy: str = "strict",
) -> dict[str, Any]:

    _require_runtime()
    service_policy = _validate_service_policy(service_policy)
    acc = _init_mode_accumulator()
    prebooking_requests, realtime_requests = requests_by_type(requests)
    loops = _build_loop_all(graph)
    loads: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    used_cycles: set[tuple[int, int]] = set()

    def insert_request(request: TripRequest) -> bool:
        loop = _nearest_loop_for_request(request, loops, graph)
        candidates: list[dict[str, Any]] = []

        for boarding_stop, boarding_index in loop.fixed_stop_indices.items():
            origin_walk = distance_shortpath(request.origin, boarding_stop, graph)
            for alighting_stop, alighting_index in loop.fixed_stop_indices.items():
                destination_walk = distance_shortpath(
                    request.destination,
                    alighting_stop,
                    graph,
                )
                walk_time = float(origin_walk + destination_walk)
                onboard_time = float(
                    _circular_travel_time(
                        boarding_index,
                        alighting_index,
                        loop.length,
                    )
                )

                for vehicle_id, offset in loop.vehicle_offsets.items():
                    first_pass = (boarding_index - offset) % loop.length
                    boarding_time = _next_cyclic_pass(
                        request.departure_time,
                        first_pass,
                        loop.length,
                    )
                    wait_time = float(boarding_time - request.departure_time)
                    if not _check_loop_capacity(
                        fleet,
                        loads,
                        vehicle_id,
                        boarding_time,
                        boarding_index,
                        alighting_index,
                        loop.length,
                    ):
                        continue

                    cycle_start_time = boarding_time - first_pass
                    added_cycles = {
                        (vehicle_id, cycle_start_time)
                    } - used_cycles
                    candidate_departures = acc.total_departures + len(added_cycles)
                    candidate_travel_distance = (
                        acc.total_travel_distance
                        + float(loop.length * len(added_cycles))
                    )
                    candidate_expenditure = _calculate_net_expenditure(
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
                            "added_cycles": added_cycles,
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

        best_choice, _ = _minimize_objective(
            candidates,
            (_build_expenditure_constraint(benchmark_expenditure),),
        )
        if best_choice is None:
            return False

        _reserve_loop_capacity(
            loads,
            int(best_choice["vehicle_id"]),
            int(best_choice["boarding_time"]),
            int(best_choice["boarding_index"]),
            int(best_choice["alighting_index"]),
            int(best_choice["route_length"]),
        )

        used_cycles.update(best_choice["added_cycles"])
        acc.served_requests += 1
        acc.total_wait += float(best_choice["wait_time"])
        acc.total_walk += float(best_choice["walk_time"])
        acc.total_onboard += float(best_choice["onboard_time"])
        _set_operator_metrics(
            acc,
            float(best_choice["candidate_travel_distance"]),
            int(best_choice["candidate_departures"]),
        )
        return True

    for request in prebooking_requests:
        if not insert_request(request):
            return _finalize_nonbaseline_mode(
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

    return _finalize_nonbaseline_mode(
        mode_id=4,
        scenario=scenario,
        requests=requests,
        benchmark_expenditure=benchmark_expenditure,
        acc=acc,
        feasible=True,
        feasibility_reason=_request_type_mode_reason(acc.served_requests, len(requests)),
    )


def _node_numeric_pair(node: GridNode, graph: nx.Graph) -> tuple[float, float]:
    if isinstance(node, tuple) and len(node) >= 2:
        try:
            return (float(node[0]), float(node[1]))
        except (TypeError, ValueError):
            pass

    position = graph.nodes[node].get("pos") if node in graph else None
    if isinstance(position, (tuple, list)) and len(position) >= 2:
        return (float(position[0]), float(position[1]))

    ordinal_by_node = {
        graph_node: index
        for index, graph_node in enumerate(sorted(graph.nodes, key=_node_sort_key))
    }
    return (float(ordinal_by_node.get(node, len(ordinal_by_node))), 0.0)


def _request_sort_key(
    request: TripRequest,
    graph: nx.Graph,
) -> tuple[int, float, float, float, float, int]: # 定义一个排序键函数，用于根据请求的departure_time、origin坐标和destination坐标对请求进行排序
    origin_x, origin_y = _node_numeric_pair(request.origin, graph)
    destination_x, destination_y = _node_numeric_pair(request.destination, graph)
    return (
        request.departure_time,
    )


# 使用DBSCAN算法对请求进行聚类，
def _dbscan_clusters(
    requests: list[TripRequest],
    graph: nx.Graph,
) -> list[list[TripRequest]]: 
    # 基于请求间的departure_time、origin网络距离、destination网络距离进行聚类

    ordered_requests = sorted(requests, key=lambda request: _request_sort_key(request, graph)) 
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
        key=lambda cluster: _request_sort_key(cluster[0], graph),
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
            origin_distance = distance_shortpath(
                request_i.origin,
                request_j.origin,
                graph,
            )
            destination_distance = distance_shortpath(
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
        if request.request_type == "pre_booking":
            prebooking_requests.append(request)
        else:
            realtime_requests.append(request)
    return prebooking_requests, realtime_requests


def _request_type_mode_reason(served_requests: int, total_requests: int) -> str: # 根据服务政策和服务的请求数量，确定模式的可行性原因
    if served_requests < total_requests:
        return "partial_service"
    return "feasible"


def _expand_route(  # 将路线的停靠点扩展为完整的节点序列，沿着图中的最短路径连接每对连续停靠点
    graph: nx.Graph,
    ordered_stops: tuple[GridNode, ...],
) -> tuple[GridNode, ...]: 
    route_nodes: list[GridNode] = []
    for index, stop in enumerate(ordered_stops):
        next_stop = ordered_stops[(index + 1) % len(ordered_stops)]
        segment = nx.shortest_path(graph, stop, next_stop, weight="weight")
        if not route_nodes:
            route_nodes.extend(segment)
        else:
            route_nodes.extend(segment[1:])
    return tuple(route_nodes)


def _node_sort_key(node: GridNode) -> tuple[str, str]: # 定义一个排序键函数，用于根据节点类型和表示形式对节点进行排序
    return (type(node).__name__, repr(node))


def _runtime_hub() -> GridNode: # 获取运行时的枢纽位置，如果network_context中定义了hub，则使用它，否则使用nets.hub
    if network_context is not None:
        return network_context.hub
    return nets.hub


def _runtime_routes() -> tuple[Any, ...]: # 获取运行时的路线定义，如果network_context中定义了routes，则使用它，否则构造一个默认的Route对象，包含nets.fixed_stops作为停靠点
    if network_context is not None:
        return network_context.routes
    return (type("Route", (), {"route_id": "route_1", "stops": tuple(nets.fixed_stops)})(),)


def _split_route_vehicle_ids(route_count: int, vehicle_count: int) -> tuple[tuple[int, ...], ...]: # 将车辆分配给各条路线
    base_count, remainder = divmod(vehicle_count, route_count)
    route_vehicle_ids = []
    next_vehicle_id = 0
    for route_index in range(route_count):
        count = base_count + int(route_index < remainder)
        ids = tuple(range(next_vehicle_id, next_vehicle_id + count))
        route_vehicle_ids.append(ids)
        next_vehicle_id += count
    return tuple(route_vehicle_ids)


def _build_loop_sub(route, route_vehicle_ids: tuple[int, ...], graph: nx.Graph) -> LoopContext: # 构建单条路线的LoopContext
    nodes = _expand_route(graph, tuple(route.stops))
    route_positions = {node: index for index, node in enumerate(nodes[:-1])}
    optional_anchor_indices: dict[GridNode, int] = {}
    route_set = set(nodes[:-1])

    for node in nodes[:-1]:
        anchor_index = route_positions[node]
        for neighbor in sorted(graph.neighbors(node), key=_node_sort_key):
            if neighbor in route_set:
                continue
            current_index = optional_anchor_indices.get(neighbor)
            if current_index is None or anchor_index < current_index:
                optional_anchor_indices[neighbor] = anchor_index
    #TODO
    length = len(nodes) - 1
    vehicle_offsets = {
        vehicle_id: (offset_index * length) // max(1, len(route_vehicle_ids))
        for offset_index, vehicle_id in enumerate(route_vehicle_ids)
    }
    loops = LoopContext(
        id=route.route_id,
        nodes=nodes,
        length=length,
        fixed_stop_indices={stop: route_positions[stop] for stop in route.stops},
        optional_stops=tuple(sorted(optional_anchor_indices, key=_node_sort_key)),
        optional_anchor_indices=optional_anchor_indices,
        vehicle_offsets=vehicle_offsets,
        headway=length / len(vehicle_offsets),
    )

    return loops


def _build_loop_all(graph: nx.Graph) -> tuple[LoopContext, ...]: # 构建所有路线的LoopContext，分配车辆，并返回一个LoopContext的元组
    routes = _runtime_routes()
    route_vehicle_ids = _split_route_vehicle_ids(len(routes), fleet.num)
    loops = tuple(
        _build_loop_sub(route, vehicle_ids, graph)
        for route, vehicle_ids in zip(routes, route_vehicle_ids)
        if vehicle_ids
    )
    if not loops:
        raise ValueError("fleet must contain at least one vehicle for route service")
    fs.draw_loops(loops)
    return loops


def _nearest_loop_for_request( # 找到与请求的起点和终点最近的LoopContext，基于 shortest-path distance（最短路径距离）计算
    request: TripRequest,
    loops: tuple[LoopContext, ...],
    graph: nx.Graph,
) -> LoopContext:
    def route_distance(loop: LoopContext) -> tuple[float, str]:
        stops = tuple(loop.fixed_stop_indices)
        origin_distance = min(distance_shortpath(request.origin, stop, graph) for stop in stops)
        destination_distance = min(
            distance_shortpath(request.destination, stop, graph)
            for stop in stops
        )
        return (float(origin_distance + destination_distance), loop.id)

    return min(loops, key=route_distance)


def _loop_departures_by_vehicle(
    vehicle_completion: dict[int, float],
    loops: tuple[LoopContext, ...],
) -> int:
    route_length_by_vehicle = {
        vehicle_id: loop.length
        for loop in loops
        for vehicle_id in loop.vehicle_offsets
    }
    return sum(
        _loop_departure_count(completion_time, route_length_by_vehicle[vehicle_id])
        for vehicle_id, completion_time in vehicle_completion.items()
    )





def _deviation_locs(
    point: GridNode,
    loop: LoopContext,
    graph: nx.Graph,
) -> list[dict[str, Any]]:
    max_deviation = _max_deviation_mode2()
    locations: list[dict[str, Any]] = []
    for anchor_index, anchor_node in enumerate(loop.nodes[:-1]):
        distance_to_anchor = distance_shortpath(point, anchor_node, graph)
        vehicle_deviation = min(distance_to_anchor, max_deviation)
        extra_walk = max(0.0, distance_to_anchor - max_deviation)
        locations.append(
            {
                "node": anchor_node,
                "anchor_index": anchor_index,
                "distance_to_anchor": float(distance_to_anchor),
                "vehicle_deviation": float(vehicle_deviation),
                "extra_walk": float(extra_walk),
            }
        )
    return locations





def _next_cyclic_pass(earliest_time: int, first_pass: int, cycle_length: int) -> int:
    if earliest_time <= first_pass:
        return first_pass
    cycles_needed = math.ceil((earliest_time - first_pass) / cycle_length)  # 计算从first_pass开始，达到或超过earliest_time所需的完整周期数
    return first_pass + cycles_needed * cycle_length


def _circular_travel_time(start_index: int, end_index: int, cycle_length: int) -> int:
    delta = (end_index - start_index) % cycle_length
    return delta if delta > 0 else cycle_length


def _check_loop_capacity(fleet, 
    loads: defaultdict[tuple[int, int, float], int],
    vehicle_id: int,
    route_start_time: float,
    boarding_index: int,
    alighting_index: int,
    route_length: int,
) -> bool:
    travel_time = _circular_travel_time(boarding_index, alighting_index, route_length)
    for step in range(travel_time):
        edge_index = (boarding_index + step) % route_length
        edge_time = route_start_time + step
        if loads[(vehicle_id, edge_index, edge_time)] >= fleet.cap:
            return False
    return True


def _reserve_loop_capacity( # 在loads数据结构中为指定车辆和路线段的时间段预留容量，基于boarding_index和alighting_index计算旅行时间，并在每个相关的时间步增加负载计数
    loads: defaultdict[tuple[int, int, float], int],
    vehicle_id: int,
    route_start_time: float,
    boarding_index: int,
    alighting_index: int,
    route_length: int,
) -> None:
    travel_time = _circular_travel_time(boarding_index, alighting_index, route_length)
    for step in range(travel_time):
        edge_index = (boarding_index + step) % route_length
        edge_time = route_start_time + step
        loads[(vehicle_id, edge_index, edge_time)] += 1





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
    feasible: bool, #布尔值，表示方案是否可行
    feasibility_reason: str,
    total_trips: int | None = None,
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
        "hs": scenario["hs"],
        "ht": scenario["ht"],
        "seed": scenario["seed"],
        "replication_id": scenario.get("replication_id"),
        "fleet_size": int(scenario.get("fleet_size", fleet.num)),
        "capacity": int(scenario.get("capacity", fleet.cap)),
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
        "max_concurrent_trips": (
            None if max_concurrent_trips is None else int(max_concurrent_trips)
        ),
        "vehicle_reuse_ratio": _round_metric(vehicle_reuse_ratio),
    }


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)








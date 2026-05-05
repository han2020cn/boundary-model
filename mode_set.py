from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable
import math

import networkx as nx

from demand_generation import TripRequest

# parameters
DISTANCE_CACHE_KEY = "_boundary_model_shortest_path_lengths"
GridNode = tuple[int, int]
Scenario = dict[str, Any]
CandidateConstraint = Callable[[dict[str, Any]], str | None]
# 如果需要成组传递车辆参数，可以创建class
FLEET_SIZE = 7
VEHICLE_CAPACITY = 30
HUB: GridNode = (4, 4)
FIXED_STOPS: tuple[GridNode, ...] = (
    (1, 5),
    (1, 7),
    (5, 7),
    (9, 7),
    (9, 5),
    (9, 3),
    (5, 3),
    (1, 3),
)
MODE_NAMES = {
    1: "fixed_route",
    2: "deviated_route",
    3: "drt_rolling_horizon",
    4: "hub_and_spoke",
}

SPOKE_ORDER = ("north", "east", "south", "west")
RESULT_COLUMNS = [
    "scenario_id",
    "lambda",
    "hs",
    "ht",
    "seed",
    "mode_id",
    "mode_name",
    "feasible",
    "feasibility_reason",
    "total_requests",
    "served_requests",
    "unserved_requests",
    "benchmark_expenditure",
    "net_expenditure", # cost
    "total_wait",
    "total_walk",
    "total_onboard",
    "total_service_time",
    "avg_wait",
    "avg_walk",
    "avg_onboard",
    "avg_service_time",
    "fleet_size",
    "capacity",
]

# loop route for mode 1 and 2
@dataclass(frozen=True, slots=True)
class LoopContext:
    route_nodes: tuple[GridNode, ...]
    route_length: int
    fixed_stop_indices: dict[GridNode, int]
    optional_stops: tuple[GridNode, ...]
    optional_anchor_indices: dict[GridNode, int]
    vehicle_offsets: tuple[int, ...]

# spoke paths for mode 4
@dataclass(frozen=True, slots=True)
class SpokeVehicle:
    vehicle_id: int
    spoke_name: str
    first_departure: int

# results
@dataclass
class requests:
    request_id: int
    origin: GridNode
    destination: GridNode
    departure_time: int

@dataclass(slots=True)
class ModeAccumulator:
    served_requests: int = 0
    total_wait: float = 0.0
    total_walk: float = 0.0
    total_onboard: float = 0.0
    total_travel_distance: float = 0.0
    total_departures: int = 0
    net_expenditure: float = 0.0


@dataclass(frozen=True, slots=True)
class DrtEvent:
    request: TripRequest
    event_type: str


@dataclass(slots=True)
class DrtVehicleState:
    current_time: int = 0
    current_location: GridNode = HUB
    pending_events: list[DrtEvent] = field(default_factory=list)
    pending_evaluation: dict[str, Any] = field(default_factory=dict)
    onboard_requests: dict[int, TripRequest] = field(default_factory=dict)
    onboard_pickup_times: dict[int, float] = field(default_factory=dict)
    committed_wait: float = 0.0
    committed_onboard: float = 0.0
    committed_active_travel: float = 0.0
    has_departed: bool = False


def build_grid_graph(grid_size: int) -> nx.Graph:
    graph = nx.grid_2d_graph(grid_size, grid_size)
    nx.set_edge_attributes(graph, 1, "weight")
    return graph


def manhattan_distance(
    a: GridNode,
    b: GridNode,
    net_graph: nx.Graph | None = None,
) -> int:
    if net_graph is None:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

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

    return int(distance_cache[a][b])


def _init_mode_accumulator() -> ModeAccumulator:
    return ModeAccumulator()


def _calculate_net_expenditure( # 计算净支出 
    total_travel_distance: float,
    total_departures: int,
    served_requests: int,
) -> float:
    return float(total_travel_distance * 1.5 + total_departures * 100 - served_requests)


def _set_operator_metrics(
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


def _loop_departure_count(completion_time: float, cycle_length: int) -> int:
    if completion_time <= 0.0:
        return 0
    return int(math.ceil(completion_time / cycle_length))


def _finalize_nonbaseline_mode(
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
    )


def _minimize_candidate(
    candidates: Iterable[dict[str, Any]],
    constraints: Iterable[CandidateConstraint],
) -> tuple[dict[str, Any] | None, set[str]]:
    failure_reasons: set[str] = set()
    ordered_constraints = tuple(constraints)

    for candidate in sorted(candidates, key=lambda item: item["ranking"]):
        for constraint in ordered_constraints:
            failure_reason = constraint(candidate)
            if failure_reason is not None:
                failure_reasons.add(failure_reason)
                break
        else:
            return candidate, failure_reasons

    return None, failure_reasons


def _validate_service_policy(service_policy: str) -> str:
    if service_policy not in {"strict", "skip"}:
        raise ValueError("service_policy must be 'strict' or 'skip'")
    return service_policy


def _mode_result_reason(
    served_requests: int,
    total_requests: int,
    service_policy: str,
) -> str:
    if service_policy == "skip" and served_requests < total_requests:
        return "partial_service"
    return "feasible"


def _build_loop_capacity_constraint(
    loads: defaultdict[tuple[int, int, int], int],
    route_length: int,
) -> CandidateConstraint:
    def constraint(candidate: dict[str, Any]) -> str | None:
        if _check_loop_capacity(
            loads,
            int(candidate["vehicle_id"]),
            int(candidate["route_start_time"]),
            int(candidate["boarding_anchor"]),
            int(candidate["alighting_anchor"]),
            route_length,
        ):
            return None
        return "capacity_limit"

    return constraint


def _build_drt_capacity_constraint() -> CandidateConstraint:
    def constraint(candidate: dict[str, Any]) -> str | None:
        if bool(candidate["evaluation"].get("capacity_feasible", False)):
            return None
        return "capacity_limit"

    return constraint


def _build_path_capacity_constraint(
    loads: defaultdict[tuple[int, GridNode, GridNode, int], int],
) -> CandidateConstraint:
    def constraint(candidate: dict[str, Any]) -> str | None:
        if _check_path_capacity(
            loads,
            int(candidate["vehicle_id"]),
            list(candidate["path"]),
            int(candidate["start_time"]),
        ):
            return None
        return "capacity_limit"

    return constraint


def evaluate_mode_1( # 评估固定路线模式
    requests: list[TripRequest],
    scenario: Scenario,
    graph: nx.Graph,
) -> dict[str, Any]:
    loop = _build_loop_context(graph)
    loads: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    vehicle_completion = {vehicle_id: 0.0 for vehicle_id in range(FLEET_SIZE)}
    served_requests = 0
    total_wait = 0.0
    total_walk = 0.0
    total_onboard = 0.0
    #对每个request循环
    for request in _sorted_requests(requests):
        best_choice: dict[str, Any] | None = None

        for boarding_stop, boarding_index in loop.fixed_stop_indices.items():
            origin_walk = manhattan_distance(request.origin, boarding_stop, graph)
            for alighting_stop, alighting_index in loop.fixed_stop_indices.items():
                destination_walk = manhattan_distance(request.destination, alighting_stop, graph)
                walk_time = float(origin_walk + destination_walk)
                onboard_time = float(
                    _circular_travel_time(
                        boarding_index,
                        alighting_index,
                        loop.route_length,
                    )
                )

                for vehicle_id, offset in enumerate(loop.vehicle_offsets):
                    first_pass = (boarding_index - offset) % loop.route_length
                    boarding_time = _next_cyclic_pass(
                        request.departure_time,
                        first_pass,
                        loop.route_length,
                    )
                    wait_time = float(boarding_time - request.departure_time)
                    if not _check_loop_capacity(
                        loads,
                        vehicle_id,
                        boarding_time,
                        boarding_index,
                        alighting_index,
                        loop.route_length,
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
                    if best_choice is None or ranking < best_choice["ranking"]:
                        best_choice = {
                            "vehicle_id": vehicle_id,
                            "boarding_index": boarding_index,
                            "alighting_index": alighting_index,
                            "boarding_time": boarding_time,
                            "operator_finish": operator_finish,
                            "wait_time": wait_time,
                            "walk_time": walk_time,
                            "onboard_time": onboard_time,
                            "ranking": ranking,
                        }

        if best_choice is None:
            total_travel_distance = float(sum(vehicle_completion.values()))
            total_departures = sum(
                _loop_departure_count(completion_time, loop.route_length)
                for completion_time in vehicle_completion.values()
            )
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
            loop.route_length,
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
    total_departures = sum(
        _loop_departure_count(completion_time, loop.route_length)
        for completion_time in vehicle_completion.values()
    )
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


def evaluate_mode_2( # 评估偏离路线模式 deviated route
    requests: list[TripRequest],
    scenario: Scenario,
    graph: nx.Graph,
    benchmark_expenditure: float | None,
    service_policy: str = "strict",
) -> dict[str, Any]:

    service_policy = _validate_service_policy(service_policy)
    acc = _init_mode_accumulator()
    prebooking_requests, realtime_requests = _partition_requests_by_type(requests)
    loop = _build_loop_context(graph)
    candidate_locations = _build_mode_2_locations(loop)
    loads: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    vehicle_completion = {vehicle_id: 0.0 for vehicle_id in range(FLEET_SIZE)}
    vehicle_delay = {vehicle_id: 0 for vehicle_id in range(FLEET_SIZE)}

    def insert_request(request: TripRequest) -> bool:
        candidates: list[dict[str, Any]] = []

        for boarding_location in candidate_locations:
            boarding_node = boarding_location["node"]
            boarding_anchor = int(boarding_location["anchor_index"])
            boarding_optional = bool(boarding_location["optional"])
            origin_walk = manhattan_distance(request.origin, boarding_node, graph)

            for alighting_location in candidate_locations:
                alighting_node = alighting_location["node"]
                alighting_anchor = int(alighting_location["anchor_index"])
                alighting_optional = bool(alighting_location["optional"])
                destination_walk = manhattan_distance(request.destination, alighting_node, graph)
                walk_time = float(origin_walk + destination_walk)
                base_route_time = _circular_travel_time(
                    boarding_anchor,
                    alighting_anchor,
                    loop.route_length,
                )
                passenger_onboard = float(
                    base_route_time
                    + int(boarding_optional)
                    + int(alighting_optional)
                )
                optional_count = int(boarding_optional) + int(alighting_optional)
                route_entry_offset = 2 if boarding_optional else 0
                passenger_board_offset = 1 if boarding_optional else 0
                operator_finish_offset = route_entry_offset + base_route_time + (
                    2 if alighting_optional else 0
                )

                for vehicle_id, offset in enumerate(loop.vehicle_offsets):
                    delayed_first_pass = (
                        (boarding_anchor - offset) % loop.route_length
                    ) + vehicle_delay[vehicle_id]
                    anchor_time = _next_cyclic_pass(
                        request.departure_time,
                        delayed_first_pass,
                        loop.route_length,
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
                        - _loop_departure_count(vehicle_completion[vehicle_id], loop.route_length)
                        + _loop_departure_count(candidate_completion, loop.route_length)
                    )
                    candidate_expenditure = _calculate_net_expenditure(
                        candidate_travel_distance,
                        candidate_departures,
                        acc.served_requests + 1,
                    )

                    ranking = (
                        wait_time + walk_time + passenger_onboard,
                        wait_time,
                        walk_time,
                        passenger_onboard,
                        optional_count,
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
                            "added_delay": 2 * optional_count,
                            "wait_time": wait_time,
                            "walk_time": walk_time,
                            "onboard_time": passenger_onboard,
                            "ranking": ranking,
                        }
                    )

        best_choice, _ = _minimize_candidate(
            candidates,
            (_build_loop_capacity_constraint(loads, loop.route_length),),
        )

        if best_choice is None:
            return False

        _reserve_loop_capacity(
            loads,
            int(best_choice["vehicle_id"]),
            int(best_choice["route_start_time"]),
            int(best_choice["boarding_anchor"]),
            int(best_choice["alighting_anchor"]),
            loop.route_length,
        )
        vehicle_id = int(best_choice["vehicle_id"])
        vehicle_completion[vehicle_id] = float(best_choice["candidate_completion"])
        vehicle_delay[vehicle_id] += int(best_choice["added_delay"])
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


def evaluate_mode_3( # 评估动态路线模式 DRT rolling horizon **lookahead = 20** 
    requests: list[TripRequest],
    scenario: Scenario,
    graph: nx.Graph,
    benchmark_expenditure: float | None,
    service_policy: str = "strict",
) -> dict[str, Any]:

    service_policy = _validate_service_policy(service_policy)
    acc = _init_mode_accumulator()
    prebooking_requests, realtime_requests = _partition_requests_by_type(requests)
    vehicle_states: dict[int, DrtVehicleState] = {
        vehicle_id: DrtVehicleState() for vehicle_id in range(FLEET_SIZE)
    }
    for state in vehicle_states.values():
        state.pending_evaluation = _evaluate_drt_event_schedule([], graph)

    scheduled_request_ids: set[int] = set()
    skipped_request_ids: set[int] = set()
    lookahead = 20 # 滚动规划的时间窗口大小 (rolling horizon lookahead), 每次规划时只考虑在当前时间加上lookahead内的请求
    step = 10
    max_departure = max((request.departure_time for request in realtime_requests), default=0)
    planning_time = 0 # 从0开始的滚动规划时间, 每次循环增加step，直到超过max_departure + lookahead或者所有请求都被安排
    sorted_realtime_requests = _sorted_requests(realtime_requests)

    def sync_accumulator() -> None:
        totals = _drt_state_totals(vehicle_states)
        acc.served_requests = len(scheduled_request_ids)
        acc.total_wait = totals["wait"]
        acc.total_walk = 0.0
        acc.total_onboard = totals["onboard"]
        _set_operator_metrics(
            acc,
            totals["travel_distance"],
            int(totals["departures"]),
        )

    def insert_request(request: TripRequest) -> bool:
        current_totals = _drt_state_totals(vehicle_states)
        current_total_service = current_totals["wait"] + current_totals["onboard"]
        candidates: list[dict[str, Any]] = []
        for vehicle_id, state in vehicle_states.items():
            schedule = state.pending_events
            current_pending_wait = _sum_evaluation_metric(
                state.pending_evaluation,
                "wait",
            )
            current_pending_onboard = _sum_evaluation_metric(
                state.pending_evaluation,
                "onboard",
            )
            current_pending_travel = float(state.pending_evaluation["active_travel"])
            pickup_event = DrtEvent(request=request, event_type="pickup")
            dropoff_event = DrtEvent(request=request, event_type="dropoff")
            for pickup_position in range(len(schedule) + 1):
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
                    candidate_evaluation = _evaluate_drt_event_schedule(
                        candidate_schedule,
                        graph,
                        start_time=state.current_time,
                        start_location=state.current_location,
                        onboard_requests=state.onboard_requests,
                        onboard_pickup_times=state.onboard_pickup_times,
                    )
                    candidate_pending_wait = _sum_evaluation_metric(
                        candidate_evaluation,
                        "wait",
                    )
                    candidate_pending_onboard = _sum_evaluation_metric(
                        candidate_evaluation,
                        "onboard",
                    )
                    candidate_total_service = (
                        current_total_service
                        - current_pending_wait
                        - current_pending_onboard
                        + candidate_pending_wait
                        + candidate_pending_onboard
                    )
                    candidate_total_wait = (
                        current_totals["wait"]
                        - current_pending_wait
                        + candidate_pending_wait
                    )
                    candidate_total_travel = (
                        current_totals["travel_distance"]
                        - current_pending_travel
                        + float(candidate_evaluation["active_travel"])
                    )
                    candidate_departures = int(
                        current_totals["departures"] + int(not state.has_departed)
                    )
                    candidate_expenditure = _calculate_net_expenditure(
                        candidate_total_travel,
                        candidate_departures,
                        len(scheduled_request_ids) + 1,
                    )

                    ranking = (
                        candidate_total_service,
                        candidate_total_wait,
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
                            "ranking": ranking,
                        }
                    )

        best_insertion, _ = _minimize_candidate(
            candidates,
            (_build_drt_capacity_constraint(),),
        )

        if best_insertion is None:
            return False

        vehicle_id = int(best_insertion["vehicle_id"])
        vehicle_states[vehicle_id].pending_events = list(best_insertion["schedule"])
        vehicle_states[vehicle_id].pending_evaluation = dict(best_insertion["evaluation"])
        vehicle_states[vehicle_id].has_departed = True
        scheduled_request_ids.add(request.request_id)
        return True

    def processed_realtime_count() -> int:
        return sum(
            1
            for request in realtime_requests
            if request.request_id in scheduled_request_ids
            or request.request_id in skipped_request_ids
        )

    for request in prebooking_requests:
        if not insert_request(request):
            sync_accumulator()
            return _finalize_nonbaseline_mode(
                mode_id=3,
                scenario=scenario,
                requests=requests,
                benchmark_expenditure=benchmark_expenditure,
                acc=acc,
                feasible=False,
                feasibility_reason="prebooking_insertion_failed",
            )

    while (
        planning_time <= max_departure + lookahead
        and processed_realtime_count() < len(realtime_requests)
    ):
        for state in vehicle_states.values():
            _advance_drt_vehicle_state(state, planning_time, graph)

        visible_requests = [
            request
            for request in sorted_realtime_requests
            if request.request_id not in scheduled_request_ids
            and request.request_id not in skipped_request_ids
            and request.departure_time <= planning_time + lookahead
        ]

        for request in visible_requests:
            if request.request_id in scheduled_request_ids:
                continue

            if not insert_request(request):
                sync_accumulator()
                skipped_request_ids.add(request.request_id)

        planning_time += step

    for request in realtime_requests:
        if (
            request.request_id not in scheduled_request_ids
            and request.request_id not in skipped_request_ids
        ):
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


def evaluate_mode_4( # 评估枢纽辐射模式 hub-and-spoke
    requests: list[TripRequest],
    scenario: Scenario,
    graph: nx.Graph,
    benchmark_expenditure: float | None,
    service_policy: str = "strict",
) -> dict[str, Any]:

    service_policy = _validate_service_policy(service_policy)
    acc = _init_mode_accumulator()
    prebooking_requests, realtime_requests = _partition_requests_by_type(requests)
    spoke_paths = _build_spoke_paths(graph)
    spoke_stops = _build_spoke_stop_list(spoke_paths)
    dispatches = _build_spoke_dispatches()
    loads: defaultdict[tuple[int, GridNode, GridNode, int], int] = defaultdict(int)
    used_cycles: set[tuple[int, int]] = set()

    def insert_request(request: TripRequest) -> bool:
        origin_stop = _nearest_spoke_stop(request.origin, spoke_stops, graph)
        destination_stop = _nearest_spoke_stop(request.destination, spoke_stops, graph)
        walk_time = float(
            manhattan_distance(request.origin, origin_stop, graph)
            + manhattan_distance(request.destination, destination_stop, graph)
        )
        # 首先尝试origin->destination的单程，如果不行再尝试origin->hub->destination的两程
        inbound_leg = _select_inbound_leg(
            origin_stop,
            request.departure_time,
            graph,
            dispatches,
            loads,
        )
        if inbound_leg is None:
            return False
        #
        outbound_leg = _select_outbound_leg(
            destination_stop,
            int(inbound_leg["arrival_time"]),
            graph,
            dispatches,
            loads,
        )
        if outbound_leg is None:
            return False

        candidate_cycles = {
            tuple(leg["departure_key"])
            for leg in (inbound_leg, outbound_leg)
            if leg.get("departure_key") is not None
        }
        added_cycles = candidate_cycles - used_cycles
        candidate_departures = acc.total_departures + len(added_cycles)
        candidate_travel_distance = acc.total_travel_distance + 8.0 * len(added_cycles)
        candidate_expenditure = _calculate_net_expenditure(
            candidate_travel_distance,
            candidate_departures,
            acc.served_requests + 1,
        )

        for leg in (inbound_leg, outbound_leg):
            if leg.get("vehicle_id") is None:
                continue
            _reserve_path_capacity(
                loads,
                int(leg["vehicle_id"]),
                list(leg["path"]),
                int(leg["start_time"]),
            )

        used_cycles.update(added_cycles)
        acc.served_requests += 1
        acc.total_wait += float(inbound_leg["wait_time"]) + float(outbound_leg["wait_time"])
        acc.total_walk += walk_time
        acc.total_onboard += float(inbound_leg["onboard_time"]) + float(outbound_leg["onboard_time"])
        _set_operator_metrics(
            acc,
            candidate_travel_distance,
            candidate_departures,
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


def _sorted_requests(requests: list[TripRequest]) -> list[TripRequest]:
    return sorted(requests, key=lambda request: (request.departure_time, request.request_id))


def _partition_requests_by_type(
    requests: list[TripRequest],
) -> tuple[list[TripRequest], list[TripRequest]]:
    prebooking_requests: list[TripRequest] = []
    realtime_requests: list[TripRequest] = []
    for request in _sorted_requests(requests):
        if request.request_type == "pre_booking":
            prebooking_requests.append(request)
        else:
            realtime_requests.append(request)
    return prebooking_requests, realtime_requests


def _request_type_mode_reason(served_requests: int, total_requests: int) -> str:
    if served_requests < total_requests:
        return "partial_service"
    return "feasible"


def _expand_route(
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


def _build_loop_context(graph: nx.Graph) -> LoopContext:
    route_nodes = _expand_route(graph, FIXED_STOPS)
    route_positions = {node: index for index, node in enumerate(route_nodes[:-1])}
    optional_anchor_indices: dict[GridNode, int] = {}
    route_set = set(route_nodes[:-1])

    for node in route_nodes[:-1]:
        anchor_index = route_positions[node]
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor in route_set:
                continue
            current_index = optional_anchor_indices.get(neighbor)
            if current_index is None or anchor_index < current_index:
                optional_anchor_indices[neighbor] = anchor_index

    route_length = len(route_nodes) - 1
    return LoopContext(
        route_nodes=route_nodes,
        route_length=route_length,
        fixed_stop_indices={stop: route_positions[stop] for stop in FIXED_STOPS},
        optional_stops=tuple(sorted(optional_anchor_indices)),
        optional_anchor_indices=optional_anchor_indices,
        vehicle_offsets=tuple((vehicle_id * route_length) // FLEET_SIZE for vehicle_id in range(FLEET_SIZE)),
    )


def _build_mode_2_locations(loop: LoopContext) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    for stop, anchor_index in loop.fixed_stop_indices.items():
        locations.append(
            {
                "node": stop,
                "anchor_index": anchor_index,
                "optional": False,
            }
        )
    for stop in loop.optional_stops:
        locations.append(
            {
                "node": stop,
                "anchor_index": loop.optional_anchor_indices[stop],
                "optional": True,
            }
        )
    return locations


def _next_cyclic_pass(earliest_time: int, first_pass: int, cycle_length: int) -> int:
    if earliest_time <= first_pass:
        return first_pass
    cycles_needed = math.ceil((earliest_time - first_pass) / cycle_length)
    return first_pass + cycles_needed * cycle_length


def _circular_travel_time(start_index: int, end_index: int, cycle_length: int) -> int:
    delta = (end_index - start_index) % cycle_length
    return delta if delta > 0 else cycle_length


def _check_loop_capacity(
    loads: defaultdict[tuple[int, int, int], int],
    vehicle_id: int,
    route_start_time: int,
    boarding_index: int,
    alighting_index: int,
    route_length: int,
) -> bool:
    travel_time = _circular_travel_time(boarding_index, alighting_index, route_length)
    for step in range(travel_time):
        edge_index = (boarding_index + step) % route_length
        edge_time = route_start_time + step
        if loads[(vehicle_id, edge_index, edge_time)] >= VEHICLE_CAPACITY:
            return False
    return True


def _reserve_loop_capacity(
    loads: defaultdict[tuple[int, int, int], int],
    vehicle_id: int,
    route_start_time: int,
    boarding_index: int,
    alighting_index: int,
    route_length: int,
) -> None:
    travel_time = _circular_travel_time(boarding_index, alighting_index, route_length)
    for step in range(travel_time):
        edge_index = (boarding_index + step) % route_length
        edge_time = route_start_time + step
        loads[(vehicle_id, edge_index, edge_time)] += 1


def _evaluate_drt_event_schedule(
    scheduled_events: list[DrtEvent],
    graph: nx.Graph,
    start_time: int = 0,
    start_location: GridNode = HUB,
    onboard_requests: dict[int, TripRequest] | None = None,
    onboard_pickup_times: dict[int, float] | None = None,
) -> dict[str, Any]:
    assignments: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    current_time = start_time
    current_location = start_location
    active_travel = 0.0
    active_requests = dict(onboard_requests or {})
    pickup_times = dict(onboard_pickup_times or {})
    completed_request_ids: set[int] = set()
    capacity_feasible = len(active_requests) <= VEHICLE_CAPACITY

    for event in scheduled_events:
        request = event.request
        if event.event_type == "pickup":
            if (
                request.request_id in active_requests
                or request.request_id in completed_request_ids
            ):
                capacity_feasible = False
                break

            travel_to_origin = manhattan_distance(
                current_location,
                request.origin,
                graph,
            )
            arrival_at_origin = current_time + travel_to_origin
            event_time = max(request.departure_time, arrival_at_origin)
            wait_time = event_time - request.departure_time
            active_travel += float(travel_to_origin)
            current_time = event_time
            current_location = request.origin
            active_requests[request.request_id] = request
            pickup_times[request.request_id] = float(event_time)
            capacity_feasible = (
                capacity_feasible
                and len(active_requests) <= VEHICLE_CAPACITY
            )
            assignments.append(
                {
                    "request_id": request.request_id,
                    "pickup_time": float(event_time),
                    "dropoff_time": None,
                    "wait": float(wait_time),
                    "onboard": 0.0,
                    "active_travel_increment": float(travel_to_origin),
                }
            )
            event_records.append(
                {
                    "event": event,
                    "event_time": float(event_time),
                    "location": request.origin,
                    "wait": float(wait_time),
                    "onboard": 0.0,
                    "active_travel_increment": float(travel_to_origin),
                }
            )
            continue

        if event.event_type != "dropoff":
            capacity_feasible = False
            break
        if request.request_id not in active_requests:
            capacity_feasible = False
            break

        travel_to_destination = manhattan_distance(
            current_location,
            request.destination,
            graph,
        )
        event_time = current_time + travel_to_destination
        pickup_time = float(pickup_times[request.request_id])
        onboard_time = float(event_time - pickup_time)
        active_travel += float(travel_to_destination)
        current_time = event_time
        current_location = request.destination
        del active_requests[request.request_id]
        del pickup_times[request.request_id]
        completed_request_ids.add(request.request_id)
        assignments.append(
            {
                "request_id": request.request_id,
                "pickup_time": pickup_time,
                "dropoff_time": float(event_time),
                "wait": 0.0,
                "onboard": onboard_time,
                "active_travel_increment": float(travel_to_destination),
            }
        )
        event_records.append(
            {
                "event": event,
                "event_time": float(event_time),
                "location": request.destination,
                "wait": 0.0,
                "onboard": onboard_time,
                "active_travel_increment": float(travel_to_destination),
            }
        )

    return {
        "assignments": assignments,
        "events": event_records,
        "completion_time": float(current_time),
        "active_travel": float(active_travel),
        "completion_location": current_location,
        "capacity_feasible": bool(capacity_feasible),
    }


def _evaluate_drt_schedule(
    scheduled_requests: list[TripRequest],
    graph: nx.Graph,
    start_time: int = 0,
    start_location: GridNode = HUB,
) -> dict[str, Any]:
    scheduled_events: list[DrtEvent] = []
    for request in scheduled_requests:
        scheduled_events.append(DrtEvent(request=request, event_type="pickup"))
        scheduled_events.append(DrtEvent(request=request, event_type="dropoff"))
    return _evaluate_drt_event_schedule(
        scheduled_events,
        graph,
        start_time=start_time,
        start_location=start_location,
    )


def _sum_assignment_metric(
    vehicle_evaluations: dict[int, dict[str, Any]],
    metric: str,
) -> float:
    total = 0.0
    for evaluation in vehicle_evaluations.values():
        total += sum(float(assignment[metric]) for assignment in evaluation["assignments"])
    return total


def _sum_evaluation_metric(
    evaluation: dict[str, Any],
    metric: str,
) -> float:
    return sum(float(assignment[metric]) for assignment in evaluation["assignments"])


def _advance_drt_vehicle_state(
    state: DrtVehicleState,
    planning_time: int,
    graph: nx.Graph,
) -> None:
    if not state.pending_events:
        state.pending_evaluation = _evaluate_drt_event_schedule(
            [],
            graph,
            start_time=state.current_time,
            start_location=state.current_location,
            onboard_requests=state.onboard_requests,
            onboard_pickup_times=state.onboard_pickup_times,
        )
        return

    evaluation = _evaluate_drt_event_schedule(
        state.pending_events,
        graph,
        start_time=state.current_time,
        start_location=state.current_location,
        onboard_requests=state.onboard_requests,
        onboard_pickup_times=state.onboard_pickup_times,
    )
    committed_count = 0
    for event_record in evaluation["events"]:
        if float(event_record["event_time"]) <= planning_time:
            committed_count += 1
        else:
            break

    if committed_count == 0:
        state.pending_evaluation = evaluation
        return

    for event_record in evaluation["events"][:committed_count]:
        event = event_record["event"]
        request = event.request
        state.committed_active_travel += float(
            event_record["active_travel_increment"]
        )
        if event.event_type == "pickup":
            state.committed_wait += float(event_record["wait"])
            state.onboard_requests[request.request_id] = request
            state.onboard_pickup_times[request.request_id] = float(
                event_record["event_time"]
            )
        else:
            pickup_time = state.onboard_pickup_times.pop(request.request_id)
            state.onboard_requests.pop(request.request_id)
            state.committed_onboard += float(
                event_record["event_time"] - pickup_time
            )

    last_event_record = evaluation["events"][committed_count - 1]
    state.current_time = int(last_event_record["event_time"])
    state.current_location = last_event_record["location"]
    state.pending_events = state.pending_events[committed_count:]
    state.pending_evaluation = _evaluate_drt_event_schedule(
        state.pending_events,
        graph,
        start_time=state.current_time,
        start_location=state.current_location,
        onboard_requests=state.onboard_requests,
        onboard_pickup_times=state.onboard_pickup_times,
    )


def _drt_state_totals(
    vehicle_states: dict[int, DrtVehicleState],
) -> dict[str, float]:
    total_wait = 0.0
    total_onboard = 0.0
    total_travel_distance = 0.0
    total_departures = 0.0

    for state in vehicle_states.values():
        total_wait += state.committed_wait + _sum_evaluation_metric(
            state.pending_evaluation,
            "wait",
        )
        total_onboard += state.committed_onboard + _sum_evaluation_metric(
            state.pending_evaluation,
            "onboard",
        )
        total_travel_distance += state.committed_active_travel + float(
            state.pending_evaluation["active_travel"]
        )
        total_departures += float(state.has_departed)

    return {
        "wait": total_wait,
        "onboard": total_onboard,
        "travel_distance": total_travel_distance,
        "departures": total_departures,
    }


def _build_spoke_paths(graph: nx.Graph) -> dict[str, tuple[GridNode, ...]]:
    edge_nodes = {
        "north": (HUB[0], 8),
        "east": (8, HUB[1]),
        "south": (HUB[0], 0),
        "west": (0, HUB[1]),
    }
    return {
        name: tuple(nx.shortest_path(graph, HUB, edge_node, weight="weight"))
        for name, edge_node in edge_nodes.items()
    }


def _build_spoke_stop_list(
    spoke_paths: dict[str, tuple[GridNode, ...]],
) -> tuple[GridNode, ...]:
    all_stops = {HUB}
    for path in spoke_paths.values():
        all_stops.update(path)
    return tuple(sorted(all_stops))


def _build_spoke_dispatches() -> dict[str, list[SpokeVehicle]]:
    dispatches = {name: [] for name in SPOKE_ORDER}
    for vehicle_id in range(FLEET_SIZE):
        dispatch = SpokeVehicle(
            vehicle_id=vehicle_id,
            spoke_name=SPOKE_ORDER[vehicle_id % len(SPOKE_ORDER)],
            first_departure=vehicle_id,
        )
        dispatches[dispatch.spoke_name].append(dispatch)
    return dispatches


def _nearest_spoke_stop(
    point: GridNode,
    spoke_stops: tuple[GridNode, ...],
    graph: nx.Graph,
) -> GridNode:
    return min(
        spoke_stops,
        key=lambda stop: (
            manhattan_distance(point, stop, graph),
            abs(stop[0] - HUB[0]) + abs(stop[1] - HUB[1]),
            stop[0],
            stop[1],
        ),
    )


def _spoke_name_for_stop(stop: GridNode) -> str:
    if stop == HUB:
        return "hub"
    if stop[0] == HUB[0]:
        return "north" if stop[1] > HUB[1] else "south"
    return "east" if stop[0] > HUB[0] else "west"


def _select_inbound_leg(
    stop: GridNode,
    earliest_time: int,
    graph: nx.Graph,
    dispatches: dict[str, list[SpokeVehicle]],
    loads: defaultdict[tuple[int, GridNode, GridNode, int], int],
) -> dict[str, Any] | None:
    if stop == HUB:
        return {
            "vehicle_id": None,
            "path": [HUB],
            "start_time": float(earliest_time),
            "arrival_time": float(earliest_time),
            "wait_time": 0.0,
            "onboard_time": 0.0,
            "cycle_finish": float(earliest_time),
            "departure_key": None,
        }

    spoke_name = _spoke_name_for_stop(stop)
    path_to_hub = nx.shortest_path(graph, stop, HUB, weight="weight")
    distance_to_hub = len(path_to_hub) - 1
    candidates: list[dict[str, Any]] = []

    for dispatch in dispatches[spoke_name]:
        first_inbound_pass = dispatch.first_departure + 8 - distance_to_hub
        boarding_time = _next_cyclic_pass(earliest_time, first_inbound_pass, 8)

        arrival_time = boarding_time + distance_to_hub
        cycle_start_time = boarding_time - (8 - distance_to_hub)
        ranking = (
            (boarding_time - earliest_time) + distance_to_hub,
            boarding_time - earliest_time,
            distance_to_hub,
            arrival_time,
            dispatch.vehicle_id,
        )
        candidates.append(
            {
                "vehicle_id": dispatch.vehicle_id,
                "path": path_to_hub,
                "start_time": float(boarding_time),
                "arrival_time": float(arrival_time),
                "wait_time": float(boarding_time - earliest_time),
                "onboard_time": float(distance_to_hub),
                "cycle_finish": float(arrival_time),
                "departure_key": (dispatch.vehicle_id, cycle_start_time),
                "ranking": ranking,
            }
        )

    best_leg, _ = _minimize_candidate(
        candidates,
        (_build_path_capacity_constraint(loads),),
    )
    return best_leg


def _select_outbound_leg(
    stop: GridNode,
    earliest_hub_departure: int,
    graph: nx.Graph,
    dispatches: dict[str, list[SpokeVehicle]],
    loads: defaultdict[tuple[int, GridNode, GridNode, int], int],
) -> dict[str, Any] | None:
    if stop == HUB:
        return {
            "vehicle_id": None,
            "path": [HUB],
            "start_time": float(earliest_hub_departure),
            "arrival_time": float(earliest_hub_departure),
            "wait_time": 0.0,
            "onboard_time": 0.0,
            "cycle_finish": float(earliest_hub_departure),
            "departure_key": None,
        }

    spoke_name = _spoke_name_for_stop(stop)
    path_from_hub = nx.shortest_path(graph, HUB, stop, weight="weight")
    distance_from_hub = len(path_from_hub) - 1
    candidates: list[dict[str, Any]] = []

    for dispatch in dispatches[spoke_name]:
        departure_time = _next_cyclic_pass(earliest_hub_departure, dispatch.first_departure, 8)

        arrival_time = departure_time + distance_from_hub
        cycle_finish = departure_time + 8
        ranking = (
            (departure_time - earliest_hub_departure) + distance_from_hub,
            departure_time - earliest_hub_departure,
            distance_from_hub,
            arrival_time,
            dispatch.vehicle_id,
        )
        candidates.append(
            {
                "vehicle_id": dispatch.vehicle_id,
                "path": path_from_hub,
                "start_time": float(departure_time),
                "arrival_time": float(arrival_time),
                "wait_time": float(departure_time - earliest_hub_departure),
                "onboard_time": float(distance_from_hub),
                "cycle_finish": float(cycle_finish),
                "departure_key": (dispatch.vehicle_id, departure_time),
                "ranking": ranking,
            }
        )

    best_leg, _ = _minimize_candidate(
        candidates,
        (_build_path_capacity_constraint(loads),),
    )
    return best_leg


def _check_path_capacity(
    loads: defaultdict[tuple[int, GridNode, GridNode, int], int],
    vehicle_id: int,
    path: list[GridNode],
    start_time: int,
) -> bool:
    for step in range(len(path) - 1):
        key = (vehicle_id, path[step], path[step + 1], start_time + step)
        if loads[key] >= VEHICLE_CAPACITY:
            return False
    return True


def _reserve_path_capacity(
    loads: defaultdict[tuple[int, GridNode, GridNode, int], int],
    vehicle_id: int,
    path: list[GridNode],
    start_time: int,
) -> None:
    for step in range(len(path) - 1):
        key = (vehicle_id, path[step], path[step + 1], start_time + step)
        loads[key] += 1


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
) -> dict[str, Any]:
    denominator = served_requests if served_requests > 0 else 0
    total_service_time = total_wait + total_walk + total_onboard
    avg_wait = total_wait / denominator if denominator else 0.0
    avg_walk = total_walk / denominator if denominator else 0.0
    avg_onboard = total_onboard / denominator if denominator else 0.0
    avg_service_time = total_service_time / denominator if denominator else 0.0

    return {
        "scenario_id": scenario["scenario_id"],
        "lambda": scenario["lambda"],
        "hs": scenario["hs"],
        "ht": scenario["ht"],
        "seed": scenario["seed"],
        "fleet_size": int(scenario.get("fleet_size", FLEET_SIZE)),
        "capacity": int(scenario.get("capacity", VEHICLE_CAPACITY)),
        "mode_id": mode_id,
        "mode_name": MODE_NAMES[mode_id],
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
        "avg_wait": _round_metric(avg_wait),
        "avg_walk": _round_metric(avg_walk),
        "avg_onboard": _round_metric(avg_onboard),
        "avg_service_time": _round_metric(avg_service_time),
    }


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)

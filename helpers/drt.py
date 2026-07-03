from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from helpers.config import TripRequest
from helpers.types import CandidateConstraint, GridNode


DISTANCE_CACHE_KEY = "_boundary_model_shortest_path_lengths"


def manhattan_distance(
    a: GridNode,
    b: GridNode,
    net_graph: nx.Graph | None = None,
) -> float:
    if net_graph is None:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

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


@dataclass(frozen=True, slots=True)
class DrtEvent:
    request: TripRequest
    event_type: str

@dataclass(frozen=True, slots=True)
class DrtTrip:
    trip_id: int
    vehicle_id: int
    start_time: float
    end_time: float
    start_location: GridNode
    end_location: GridNode
    request_ids: tuple[int, ...]
    active_travel: float

@dataclass(slots=True)
class DrtVehicleState:
    current_location: GridNode
    current_time: int = 0
    pending_events: list[DrtEvent] = field(default_factory=list)
    pending_evaluation: dict[str, Any] = field(default_factory=dict)
    onboard_requests: dict[int, TripRequest] = field(default_factory=dict)
    onboard_pickup_times: dict[int, float] = field(default_factory=dict)
    committed_wait: float = 0.0
    committed_onboard: float = 0.0
    committed_active_travel: float = 0.0
    has_departed: bool = False
    active_trip_id: int | None = None
    active_trip_vehicle_id: int | None = None
    active_trip_start_time: float | None = None
    active_trip_start_location: GridNode | None = None
    active_trip_start_active_travel: float = 0.0
    active_trip_request_ids: set[int] = field(default_factory=set)
    completed_trips: list[DrtTrip] = field(default_factory=list)

def _state_has_active_trip(state: DrtVehicleState) -> bool:
    return state.active_trip_id is not None


def _state_is_idle(state: DrtVehicleState) -> bool:
    return not state.pending_events and not state.onboard_requests


def event_request_ids(events: list[DrtEvent]) -> set[int]:
    return {event.request.request_id for event in events}


def trip_start_time_from_evaluation(
    evaluation: dict[str, Any],
    fallback_time: float,
) -> float:
    event_records = evaluation.get("events", ())
    if not event_records:
        return float(fallback_time)
    first_record = event_records[0]
    return float(first_record["event_time"]) - float(
        first_record.get("active_travel_increment", 0.0)
    )


def _start_drt_trip(
    state: DrtVehicleState,
    *,
    trip_id: int,
    vehicle_id: int,
    start_time: float,
    start_location: GridNode,
    request_ids: set[int],
) -> None:
    state.active_trip_id = int(trip_id)
    state.active_trip_vehicle_id = int(vehicle_id)
    state.active_trip_start_time = float(start_time)
    state.active_trip_start_location = start_location
    state.active_trip_start_active_travel = float(state.committed_active_travel)
    state.active_trip_request_ids = set(request_ids)
    state.has_departed = True


def _add_drt_trip_request_ids(
    state: DrtVehicleState,
    request_ids: set[int],
) -> None:
    state.active_trip_request_ids.update(request_ids)


def start_or_update_trip(
    vehicle_states: dict[int, DrtVehicleState],
    vehicle_id: int,
    evaluation: dict[str, Any],
    schedule: list[DrtEvent],
    next_trip_id: int,
) -> int:
    state = vehicle_states[vehicle_id]
    request_ids = event_request_ids(schedule)
    if _state_has_active_trip(state):
        _add_drt_trip_request_ids(state, request_ids)
        return next_trip_id

    _start_drt_trip(
        state,
        trip_id=next_trip_id,
        vehicle_id=vehicle_id,
        start_time=trip_start_time_from_evaluation(
            evaluation,
            state.current_time,
        ),
        start_location=state.current_location,
        request_ids=request_ids,
    )
    return next_trip_id + 1


def _project_open_drt_trip(
    state: DrtVehicleState,
    vehicle_id: int,
) -> DrtTrip | None:
    if state.active_trip_id is None:
        return None
    pending_evaluation = state.pending_evaluation or {}
    end_time = float(pending_evaluation.get("completion_time", state.current_time))
    end_location = pending_evaluation.get("completion_location", state.current_location)
    active_travel_total = float(state.committed_active_travel) + float(
        pending_evaluation.get("active_travel", 0.0)
    )
    return DrtTrip(
        trip_id=int(state.active_trip_id),
        vehicle_id=int(
            state.active_trip_vehicle_id
            if state.active_trip_vehicle_id is not None
            else vehicle_id
        ),
        start_time=float(
            state.active_trip_start_time
            if state.active_trip_start_time is not None
            else state.current_time
        ),
        end_time=end_time,
        start_location=(
            state.active_trip_start_location
            if state.active_trip_start_location is not None
            else state.current_location
        ),
        end_location=end_location,
        request_ids=tuple(sorted(state.active_trip_request_ids)),
        active_travel=max(
            0.0,
            active_travel_total - float(state.active_trip_start_active_travel),
        ),
    )


def _close_drt_trip(
    state: DrtVehicleState,
    vehicle_id: int,
    *,
    end_time: float,
    end_location: GridNode,
    active_travel_total: float,
) -> None:
    trip = _project_open_drt_trip(state, vehicle_id)
    if trip is None:
        return
    state.completed_trips.append(
        DrtTrip(
            trip_id=trip.trip_id,
            vehicle_id=trip.vehicle_id,
            start_time=trip.start_time,
            end_time=float(end_time),
            start_location=trip.start_location,
            end_location=end_location,
            request_ids=trip.request_ids,
            active_travel=max(
                0.0,
                float(active_travel_total)
                - float(state.active_trip_start_active_travel),
            ),
        )
    )
    state.active_trip_id = None
    state.active_trip_vehicle_id = None
    state.active_trip_start_time = None
    state.active_trip_start_location = None
    state.active_trip_start_active_travel = 0.0
    state.active_trip_request_ids.clear()


def _drt_trips_by_state(
    vehicle_states: dict[int, DrtVehicleState],
) -> list[DrtTrip]:
    trips: list[DrtTrip] = []
    for vehicle_id, state in vehicle_states.items():
        trips.extend(state.completed_trips)
        open_trip = _project_open_drt_trip(state, vehicle_id)
        if open_trip is not None:
            trips.append(open_trip)
    return trips


def _max_concurrent_trips(trips: list[DrtTrip]) -> int:
    timeline: list[tuple[float, int]] = []
    for trip in trips:
        timeline.append((float(trip.start_time), 1))
        timeline.append((float(trip.end_time), -1))
    concurrent = 0
    max_concurrent = 0
    for _, delta in sorted(timeline, key=lambda item: (item[0], item[1])):
        concurrent += delta
        max_concurrent = max(max_concurrent, concurrent)
    return int(max_concurrent)

def _drt_state_totals(
    vehicle_states: dict[int, DrtVehicleState],
) -> dict[str, float]:
    total_wait = 0.0
    total_onboard = 0.0
    total_travel_distance = 0.0
    total_trips = 0.0

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
        total_trips += float(
            len(state.completed_trips) + int(_state_has_active_trip(state))
        )

    return {
        "wait": total_wait,
        "onboard": total_onboard,
        "travel_distance": total_travel_distance,
        "trip": total_trips,
    }



def _evaluate_drt_event_schedule(
    scheduled_events: list[DrtEvent],
    graph: nx.Graph,
    nets: Any,
    fleet: Any,
    start_time: int = 0,
    start_location: GridNode | None = None,
    onboard_requests: dict[int, TripRequest] | None = None,
    onboard_pickup_times: dict[int, float] | None = None,
) -> dict[str, Any]:
    assignments: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    current_time = start_time
    current_location = nets.hub if start_location is None else start_location
    active_travel = 0.0
    active_requests = dict(onboard_requests or {})
    pickup_times = dict(onboard_pickup_times or {})
    completed_request_ids: set[int] = set()
    capacity_feasible = len(active_requests) <= fleet.cap

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
                and len(active_requests) <= fleet.cap
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
    nets: Any,
    fleet: Any,
    start_time: int = 0,
    start_location: GridNode | None = None,
) -> dict[str, Any]:
    scheduled_events: list[DrtEvent] = []
    for request in scheduled_requests:
        scheduled_events.append(DrtEvent(request=request, event_type="pickup"))
        scheduled_events.append(DrtEvent(request=request, event_type="dropoff"))
    return _evaluate_drt_event_schedule(
        scheduled_events,
        graph,
        nets,
        fleet,
        start_time=start_time,
        start_location=start_location,
    )


# 下面是一些辅助函数，用于DRT模式的评估和状态更新
def _sum_assignment_metric(
    vehicle_evaluations: dict[int, dict[str, Any]],
    metric: str,
) -> float:
    total = 0.0
    for evaluation in vehicle_evaluations.values():
        total += sum(float(assignment[metric]) for assignment in evaluation["assignments"])
    return total

# 由于DRT评估中等待时间和车上时间是分配在每个事件上的，所以需要一个函数来汇总这些指标
def _sum_evaluation_metric(
    evaluation: dict[str, Any],
    metric: str,
) -> float:
    return sum(float(assignment[metric]) for assignment in evaluation["assignments"])


def _build_drt_capacity_constraint(fleet: Any) -> CandidateConstraint:
    def constraint(candidate: dict[str, Any]) -> str | None:
        if bool(candidate["evaluation"].get("capacity_feasible", False)):
            return None
        return "capacity_limit"

    return constraint


def advance_vehicle_states(
    vehicle_states: dict[int, DrtVehicleState],
    planning_time: int,
    graph: nx.Graph,
    nets: Any,
    fleet: Any,
) -> None:
    for state in vehicle_states.values():
        _advance_drt_vehicle_state(
            state,
            planning_time,
            graph,
            nets,
            fleet,
        )


# 这个函数用于根据当前的DRT车辆状态和规划时间来推进车辆状态，处理已经发生的事件，并更新当前时间、位置、车上请求等信息
def _advance_drt_vehicle_state(
    state: DrtVehicleState,
    planning_time: int,
    graph: nx.Graph,
    nets: Any,
    fleet: Any,
) -> None:
    if not state.pending_events:
        state.pending_evaluation = _evaluate_drt_event_schedule(
            [],
            graph,
            nets,
            fleet,
            start_time=state.current_time,
            start_location=state.current_location,
            onboard_requests=state.onboard_requests,
            onboard_pickup_times=state.onboard_pickup_times,
        )
        return

    evaluation = _evaluate_drt_event_schedule(
        state.pending_events,
        graph,
        nets,
        fleet,
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
        nets,
        fleet,
        start_time=state.current_time,
        start_location=state.current_location,
        onboard_requests=state.onboard_requests,
        onboard_pickup_times=state.onboard_pickup_times,
    )
    if _state_is_idle(state):
        _close_drt_trip(
            state,
            -1,
            end_time=float(state.current_time),
            end_location=state.current_location,
            active_travel_total=float(state.committed_active_travel),
        )

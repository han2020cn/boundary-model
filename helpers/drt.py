from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN

from helpers.config import TripRequest
from helpers.types import GridNode
import helpers.common_rule as rule


DISTANCE_CACHE_KEY = "_boundary_model_shortest_path_lengths"
EventType = Literal["pickup", "dropoff"]
TripStatus = Literal["planned", "active", "completed"]
CandidateTarget = Literal["active_trip", "future_trip", "new_trip"]


@dataclass(frozen=True, slots=True)
class DrtPolicy:
    pickup_window: float
    departure_window_end: float
    overtime: float
    benchmark_expenditure: float
    recalculation_interval: float | None
    dbscan_eps: float
    dbscan_min_samples: int
    dbscan_time_scale: float
    dbscan_origin_scale: float
    dbscan_destination_scale: float
    tolerance: float = 1e-9

    @property
    def latest_return_time(self) -> float:
        return self.departure_window_end + self.overtime


@dataclass(frozen=True, slots=True)
class DrtContext:
    graph: nx.Graph
    hub: GridNode
    speed: float
    capacity: int
    vehicle_count: int
    policy: DrtPolicy
    expenditure_fn: Callable[[float, int, int, float], float]


@dataclass(frozen=True, slots=True)
class DrtEvent:
    request: TripRequest
    event_type: EventType

    @property
    def location(self) -> GridNode:
        if self.event_type == "pickup":
            return self.request.origin
        if self.event_type == "dropoff":
            return self.request.destination
        raise ValueError(f"invalid DRT event type: {self.event_type!r}")


@dataclass(frozen=True, slots=True)
class DrtRequestMetrics:
    request_id: int
    pickup_time: float
    dropoff_time: float
    wait: float
    onboard: float


@dataclass(frozen=True, slots=True)
class DrtEventRecord:
    event: DrtEvent
    event_index: int
    event_time: float
    location: GridNode
    load_after_event: int
    travel_distance_from_previous: float


@dataclass(frozen=True, slots=True)
class DrtTripEvaluation:
    feasible: bool
    violations: tuple[str, ...]
    departure_time: float
    return_time: float
    event_records: tuple[DrtEventRecord, ...]
    request_metrics: dict[int, DrtRequestMetrics]
    travel_distance: float
    operating_time: float
    total_wait: float
    total_onboard: float
    generalised_cost: float
    max_load: int


@dataclass(slots=True)
class DrtTrip:
    trip_id: int
    vehicle_id: int | None
    status: TripStatus
    events: list[DrtEvent]
    evaluation: DrtTripEvaluation
    locked_event_count: int = 0
    cluster_id: int | None = None
    created_by: Literal["prebooking", "realtime"] = "prebooking"

    @property
    def departure_time(self) -> float:
        return self.evaluation.departure_time

    @property
    def return_time(self) -> float:
        return self.evaluation.return_time

    @property
    def request_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.evaluation.request_metrics))

    @property
    def travel_distance(self) -> float:
        return self.evaluation.travel_distance

    @property
    def operating_time(self) -> float:
        return self.evaluation.operating_time


@dataclass(slots=True)
class DrtVehicleState:
    vehicle_id: int
    trip_ids: list[int] = field(default_factory=list)
    active_trip_id: int | None = None
    current_time: float = 0.0
    current_location: GridNode | None = None


@dataclass(slots=True)
class DrtSystemState:
    trips: dict[int, DrtTrip]
    vehicles: dict[int, DrtVehicleState]
    served_request_ids: set[int]
    rejected_requests: dict[int, str]
    next_trip_id: int = 1
    current_time: float = 0.0
    rejection_diagnostics: Counter[str] = field(default_factory=Counter)

    def allocate_trip_id(self) -> int:
        trip_id = self.next_trip_id
        self.next_trip_id += 1
        return trip_id

    def record_rejection(self, request_id: int, reason: str) -> None:
        self.rejected_requests[int(request_id)] = reason
        self.rejection_diagnostics[reason] += 1

    def ordered_trips(self) -> list[DrtTrip]:
        return sorted(
            self.trips.values(),
            key=lambda trip: (trip.departure_time, trip.return_time, trip.trip_id),
        )

    def vehicle_trips(self, vehicle_id: int) -> list[DrtTrip]:
        return sorted(
            (
                self.trips[trip_id]
                for trip_id in self.vehicles[vehicle_id].trip_ids
                if trip_id in self.trips
            ),
            key=lambda trip: (trip.departure_time, trip.return_time, trip.trip_id),
        )


@dataclass(frozen=True, slots=True)
class DrtInsertionCandidate:
    target_type: CandidateTarget
    request_id: int
    trip_id: int
    vehicle_id: int
    pickup_position: int
    dropoff_position: int
    replacement_trip: DrtTrip
    candidate_total_distance: float
    candidate_operating_time: float
    candidate_total_trips: int
    candidate_served_requests: int
    candidate_expenditure: float
    delta_wait: float
    delta_onboard: float
    delta_distance: float
    delta_expenditure: float
    new_request_wait: float
    ranking: tuple


@dataclass(frozen=True, slots=True)
class DrtPlanSummary:
    served_requests: int
    rejected_requests: int
    total_wait: float
    total_onboard: float
    total_travel_distance: float
    operating_time: float
    total_trips: int
    max_concurrent_trips: int
    vehicle_reuse_ratio: float | None
    net_expenditure: float
    constraints_satisfied: bool
    violations: tuple[str, ...]


def _config_value(config, names: Sequence[str], default):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def build_policy(config, benchmark_expenditure: float) -> DrtPolicy:
    interval = _config_value(
        config,
        ("drt_recalculation_interval", "recalculation_interval"),
        None,
    )
    policy = DrtPolicy(
        pickup_window=float(
            _config_value(config, ("drt_pickup_window", "pickup_window"), 30.0)
        ),
        departure_window_end=float(config.span),
        overtime=float(_config_value(config, ("drt_overtime", "overtime"), 30.0)),
        benchmark_expenditure=float(benchmark_expenditure),
        recalculation_interval=None if interval is None else float(interval),
        dbscan_eps=float(_config_value(config, ("drt_dbscan_eps",), 0.8)),
        dbscan_min_samples=int(
            _config_value(config, ("drt_dbscan_min_samples",), 2)
        ),
        dbscan_time_scale=float(
            _config_value(config, ("drt_dbscan_time_scale",), 30.0)
        ),
        dbscan_origin_scale=float(
            _config_value(config, ("drt_dbscan_origin_scale",), 1000.0)
        ),
        dbscan_destination_scale=float(
            _config_value(config, ("drt_dbscan_destination_scale",), 1000.0)
        ),
        tolerance=float(_config_value(config, ("drt_tolerance",), 1e-9)),
    )
    if policy.pickup_window < 0:
        raise ValueError("pickup_window must be non-negative")
    if policy.departure_window_end < 0:
        raise ValueError("departure_window_end must be non-negative")
    if policy.overtime < 0:
        raise ValueError("overtime must be non-negative")
    if policy.dbscan_eps <= 0:
        raise ValueError("drt_dbscan_eps must be positive")
    if policy.dbscan_min_samples <= 0:
        raise ValueError("drt_dbscan_min_samples must be positive")
    if min(
        policy.dbscan_time_scale,
        policy.dbscan_origin_scale,
        policy.dbscan_destination_scale,
    ) <= 0:
        raise ValueError("DRT DBSCAN scales must be positive")
    return policy


def build_context(
    *,
    network_context,
    fleet,
    policy: DrtPolicy,
    expenditure_fn: Callable[[float, int, int, float], float],
) -> DrtContext:
    if float(fleet.speed) <= 0:
        raise ValueError("invalid_vehicle_speed")
    if int(fleet.cap) <= 0:
        raise ValueError("invalid_vehicle_capacity")
    return DrtContext(
        graph=network_context.graph,
        hub=network_context.hub,
        speed=float(fleet.speed),
        capacity=int(fleet.cap),
        vehicle_count=int(fleet.num),
        policy=policy,
        expenditure_fn=expenditure_fn,
    )


def initialize_system(context: DrtContext) -> DrtSystemState:
    return DrtSystemState(
        trips={},
        vehicles={
            vehicle_id: DrtVehicleState(
                vehicle_id=vehicle_id,
                current_time=0.0,
                current_location=context.hub,
            )
            for vehicle_id in range(context.vehicle_count)
        },
        served_request_ids=set(),
        rejected_requests={},
    )


def network_distance(
    origin: GridNode,
    destination: GridNode,
    graph: nx.Graph,
) -> float:
    if origin == destination:
        return 0.0
    cache = graph.graph.setdefault(DISTANCE_CACHE_KEY, {})
    origin_cache = cache.setdefault(origin, {})
    if destination not in origin_cache:
        distance = float(
            nx.shortest_path_length(graph, origin, destination, weight="weight")
        )
        origin_cache[destination] = distance
        cache.setdefault(destination, {})[origin] = distance
    return float(origin_cache[destination])


def travel_time(distance: float, speed: float) -> float:
    return float(distance) / float(speed)


def _build_request_distance_matrix(
    requests: list[TripRequest],
    context: DrtContext,
) -> np.ndarray:
    count = len(requests)
    matrix = np.zeros((count, count), dtype=float)
    policy = context.policy
    scales = (
        policy.dbscan_time_scale,
        policy.dbscan_origin_scale,
        policy.dbscan_destination_scale,
    )
    if any(scale <= 0 for scale in scales):
        raise ValueError("DBSCAN scales must be positive")
    for left in range(count):
        request_left = requests[left]
        for right in range(left + 1, count):
            request_right = requests[right]
            distance = (
                abs(
                    float(request_left.departure_time)
                    - float(request_right.departure_time)
                )
                / policy.dbscan_time_scale
                + network_distance(
                    request_left.origin,
                    request_right.origin,
                    context.graph,
                )
                / policy.dbscan_origin_scale
                + network_distance(
                    request_left.destination,
                    request_right.destination,
                    context.graph,
                )
                / policy.dbscan_destination_scale
            )
            matrix[left, right] = distance
            matrix[right, left] = distance
    return matrix


def cluster_prebookings(
    requests: list[TripRequest],
    context: DrtContext,
) -> list[list[TripRequest]]:
    ordered = sorted(
        requests,
        key=lambda request: (request.departure_time, request.request_id),
    )
    if not ordered:
        return []
    if len(ordered) < context.policy.dbscan_min_samples:
        return [[request] for request in ordered]
    labels = DBSCAN(
        eps=context.policy.dbscan_eps,
        min_samples=context.policy.dbscan_min_samples,
        metric="precomputed",
    ).fit_predict(_build_request_distance_matrix(ordered, context))
    grouped: defaultdict[int, list[TripRequest]] = defaultdict(list)
    clusters: list[list[TripRequest]] = []
    for request, label_value in zip(ordered, labels):
        label = int(label_value)
        if label < 0:
            clusters.append([request])
        else:
            grouped[label].append(request)
    clusters.extend(grouped.values())
    return sorted(
        clusters,
        key=lambda cluster: (
            min(request.departure_time for request in cluster),
            min(request.request_id for request in cluster),
        ),
    )


def _integrity_violations(events: Sequence[DrtEvent]) -> tuple[str, ...]:
    if not events:
        return ("invalid_event_sequence",)
    pickups: dict[int, int] = {}
    dropoffs: dict[int, int] = {}
    violations: list[str] = []
    for index, event in enumerate(events):
        request_id = int(event.request.request_id)
        if event.event_type == "pickup":
            if request_id in pickups:
                violations.append("duplicate_pickup")
            else:
                pickups[request_id] = index
        elif event.event_type == "dropoff":
            if request_id in dropoffs:
                violations.append("duplicate_dropoff")
            else:
                dropoffs[request_id] = index
            if request_id not in pickups:
                violations.append("dropoff_before_pickup")
        else:
            violations.append("invalid_event_sequence")
    request_ids = pickups.keys() | dropoffs.keys()
    for request_id in request_ids:
        if request_id not in pickups or request_id not in dropoffs:
            violations.append("invalid_event_sequence")
        elif pickups[request_id] >= dropoffs[request_id]:
            violations.append("dropoff_before_pickup")
    return tuple(dict.fromkeys(violations))


def _empty_evaluation(violations: Sequence[str]) -> DrtTripEvaluation:
    return DrtTripEvaluation(
        feasible=False,
        violations=tuple(dict.fromkeys(violations)),
        departure_time=0.0,
        return_time=0.0,
        event_records=(),
        request_metrics={},
        travel_distance=0.0,
        operating_time=0.0,
        total_wait=0.0,
        total_onboard=0.0,
        generalised_cost=0.0,
        max_load=0,
    )


def evaluate_trip(
    events: Sequence[DrtEvent],
    context: DrtContext,
    *,
    earliest_departure_time: float = 0.0,
    fixed_departure_time: float | None = None,
) -> DrtTripEvaluation: 
    # integrity = _integrity_violations(events)   
    # if integrity:
    #     return _empty_evaluation(integrity)

    first_event = events[0]
    outbound_distance = network_distance(
        context.hub,
        first_event.location,
        context.graph,
    )
    preferred_departure = (
        float(first_event.request.departure_time)
        - travel_time(outbound_distance, context.speed)
    )
    departure_time = (
        float(fixed_departure_time)
        if fixed_departure_time is not None
        else max(float(earliest_departure_time), preferred_departure, 0.0)
    )
    current_time = departure_time
    current_location = context.hub
    load = 0
    max_load = 0
    travel_distance_total = 0.0
    pickup_times: dict[int, float] = {}
    dropoff_times: dict[int, float] = {}
    requests_by_id: dict[int, TripRequest] = {}
    records: list[DrtEventRecord] = []
    violations: list[str] = []

    for index, event in enumerate(events):
        leg_distance = network_distance(
            current_location,
            event.location,
            context.graph,
        )
        travel_distance_total += leg_distance
        arrival_time = current_time + travel_time(leg_distance, context.speed)
        request_id = int(event.request.request_id)
        requests_by_id.setdefault(request_id, event.request)
        if event.event_type == "pickup":
            event_time = max(float(event.request.departure_time), arrival_time)
            pickup_times[request_id] = event_time
            load += 1
            if event_time > (
                float(event.request.departure_time)
                + context.policy.pickup_window
                + context.policy.tolerance
            ):
                violations.append("pickup_window")
        else:
            event_time = arrival_time
            dropoff_times[request_id] = event_time
            load -= 1
        if load < 0 or load > context.capacity:
            violations.append("capacity_limit")
        max_load = max(max_load, load)
        records.append(
            DrtEventRecord(
                event=event,
                event_index=index,
                event_time=event_time,
                location=event.location,
                load_after_event=load,
                travel_distance_from_previous=leg_distance,
            )
        )
        current_time = event_time
        current_location = event.location

    return_leg = network_distance(current_location, context.hub, context.graph)
    travel_distance_total += return_leg
    return_time = current_time + travel_time(return_leg, context.speed)
    if departure_time > (
        context.policy.departure_window_end + context.policy.tolerance
    ):
        violations.append("departure_window")
    if return_time > context.policy.latest_return_time + context.policy.tolerance:
        violations.append("overtime_limit")
    if load != 0:
        violations.append("invalid_event_sequence")

    metrics: dict[int, DrtRequestMetrics] = {}
    for request_id, pickup_time in pickup_times.items():
        dropoff_time = dropoff_times[request_id]
        request = requests_by_id[request_id]
        metrics[request_id] = DrtRequestMetrics(
            request_id=request_id,
            pickup_time=pickup_time,
            dropoff_time=dropoff_time,
            wait=pickup_time - float(request.departure_time),
            onboard=dropoff_time - pickup_time,
        )
    total_wait = sum(item.wait for item in metrics.values())
    total_onboard = sum(item.onboard for item in metrics.values())
    unique_violations = tuple(dict.fromkeys(violations))
    return DrtTripEvaluation(
        feasible=not unique_violations,
        violations=unique_violations,
        departure_time=departure_time,
        return_time=return_time,
        event_records=tuple(records),
        request_metrics=metrics,
        travel_distance=travel_distance_total,
        operating_time=return_time - departure_time,
        total_wait=total_wait,
        total_onboard=total_onboard,
        generalised_cost= rule.time_objective(total_wait,0.0,total_onboard,),
        max_load=max_load,
    )


def iter_insertion_positions(
    event_count: int,
    *,
    minimum_pickup_position: int,
) -> Iterator[tuple[int, int]]:
    for pickup_position in range(minimum_pickup_position, event_count + 1):
        for dropoff_position in range(pickup_position + 1, event_count + 2):
            yield pickup_position, dropoff_position


def _insert_events(
    events: Sequence[DrtEvent],
    request: TripRequest,
    pickup_position: int,
    dropoff_position: int,
) -> list[DrtEvent]:
    inserted = list(events)
    inserted.insert(pickup_position, DrtEvent(request, "pickup"))
    inserted.insert(dropoff_position, DrtEvent(request, "dropoff"))
    return inserted


def _clone_trip(trip: DrtTrip, **changes) -> DrtTrip:
    values = {
        "trip_id": trip.trip_id,
        "vehicle_id": trip.vehicle_id,
        "status": trip.status,
        "events": list(trip.events),
        "evaluation": trip.evaluation,
        "locked_event_count": trip.locked_event_count,
        "cluster_id": trip.cluster_id,
        "created_by": trip.created_by,
    }
    values.update(changes)
    return DrtTrip(**values)


def schedule_prebooking_trips(
    trips: Sequence[DrtTrip],
    context: DrtContext,
) -> tuple[bool, list[DrtTrip], dict[int, list[int]]]:
    if trips and context.vehicle_count <= 0:
        return False, [], {}
    assignments: dict[int, list[int]] = {
        vehicle_id: [] for vehicle_id in range(context.vehicle_count)
    }
    vehicle_returns = {
        vehicle_id: 0.0 for vehicle_id in range(context.vehicle_count)
    }
    scheduled: list[DrtTrip] = []
    ordered = sorted(
        trips,
        key=lambda trip: (
            trip.evaluation.departure_time,
            min(
                (event.request.departure_time for event in trip.events),
                default=0,
            ),
            trip.trip_id,
        ),
    )
    for trip in ordered:
        best: tuple[tuple, DrtTrip] | None = None
        preferred_departure = trip.evaluation.departure_time
        for vehicle_id in range(context.vehicle_count):
            evaluation = evaluate_trip(
                trip.events,
                context,
                earliest_departure_time=vehicle_returns[vehicle_id],
            )
            if not evaluation.feasible:
                continue
            candidate = _clone_trip(
                trip,
                vehicle_id=vehicle_id,
                status="planned",
                evaluation=evaluation,
            )
            ranking = (
                max(0.0, evaluation.departure_time - preferred_departure),
                evaluation.return_time,
                vehicle_id,
            )
            if best is None or ranking < best[0]:
                best = (ranking, candidate)
        if best is None:
            return False, [], {}
        assigned = best[1]
        assert assigned.vehicle_id is not None
        scheduled.append(assigned)
        assignments[assigned.vehicle_id].append(assigned.trip_id)
        vehicle_returns[assigned.vehicle_id] = assigned.return_time
    return True, scheduled, assignments


def _current_totals(state: DrtSystemState, context: DrtContext) -> tuple:
    total_distance = sum(trip.travel_distance for trip in state.trips.values())
    operating_time = sum(trip.operating_time for trip in state.trips.values())
    total_trips = sum(bool(trip.events) for trip in state.trips.values())
    served = len(state.served_request_ids)
    expenditure = context.expenditure_fn(
        total_distance,
        total_trips,
        served,
        operating_time,
    )
    return total_distance, operating_time, total_trips, served, expenditure


def _candidate_from_replacement(
    *,
    state: DrtSystemState,
    old_trip: DrtTrip | None,
    replacement_trip: DrtTrip,
    request: TripRequest,
    pickup_position: int,
    dropoff_position: int,
    target_type: CandidateTarget,
    context: DrtContext,
) -> DrtInsertionCandidate | None:
    (
        current_distance,
        current_operating_time,
        current_trips,
        current_served,
        current_expenditure,
    ) = _current_totals(state, context)
    if old_trip is None:
        delta_wait = replacement_trip.evaluation.total_wait
        delta_onboard = replacement_trip.evaluation.total_onboard
        delta_distance = replacement_trip.travel_distance
        delta_operating_time = replacement_trip.operating_time
        candidate_trips = current_trips + 1
    else:
        delta_wait = (
            replacement_trip.evaluation.total_wait
            - old_trip.evaluation.total_wait
        )
        delta_onboard = (
            replacement_trip.evaluation.total_onboard
            - old_trip.evaluation.total_onboard
        )
        delta_distance = replacement_trip.travel_distance - old_trip.travel_distance
        delta_operating_time = (
            replacement_trip.operating_time - old_trip.operating_time
        )
        candidate_trips = current_trips
    candidate_distance = current_distance + delta_distance
    candidate_operating_time = current_operating_time + delta_operating_time
    candidate_served = current_served + 1
    candidate_expenditure = context.expenditure_fn(
        candidate_distance,
        candidate_trips,
        candidate_served,
        candidate_operating_time,
    )
    if candidate_expenditure > (
        context.policy.benchmark_expenditure + context.policy.tolerance
    ):
        return None
    request_metrics = replacement_trip.evaluation.request_metrics.get(
        int(request.request_id)
    )
    if request_metrics is None:
        return None
    delta_expenditure = candidate_expenditure - current_expenditure
    ranking = (
        delta_expenditure,
        rule.time_objective(delta_wait,0.0,delta_onboard),
        request_metrics.wait,
        int(replacement_trip.vehicle_id),
        replacement_trip.trip_id,
    )
    return DrtInsertionCandidate(
        target_type=target_type,
        request_id=int(request.request_id),
        trip_id=replacement_trip.trip_id,
        vehicle_id=int(replacement_trip.vehicle_id),
        pickup_position=pickup_position,
        dropoff_position=dropoff_position,
        replacement_trip=replacement_trip,
        candidate_total_distance=candidate_distance,
        candidate_operating_time=candidate_operating_time,
        candidate_total_trips=candidate_trips,
        candidate_served_requests=candidate_served,
        candidate_expenditure=candidate_expenditure,
        delta_wait=delta_wait,
        delta_onboard=delta_onboard,
        delta_distance=delta_distance,
        delta_expenditure=delta_expenditure,
        new_request_wait=request_metrics.wait,
        ranking=ranking,
    )


def _reschedule_prebookings(
    state: DrtSystemState,
    context: DrtContext,
) -> bool:
    feasible, scheduled, assignments = schedule_prebooking_trips(
        list(state.trips.values()),
        context,
    )
    if not feasible:
        return False
    state.trips = {trip.trip_id: trip for trip in scheduled}
    for vehicle_id, vehicle in state.vehicles.items():
        vehicle.trip_ids = list(assignments[vehicle_id])
    return True


def find_best_prebooking_insertion(
    state: DrtSystemState,
    request: TripRequest,
    candidate_trip_ids: Iterable[int],
    context: DrtContext,
) -> DrtInsertionCandidate | None:
    best: DrtInsertionCandidate | None = None
    for trip_id in candidate_trip_ids:
        trip = state.trips.get(int(trip_id))
        if trip is None or trip.status != "planned":
            continue
        for pickup_position, dropoff_position in iter_insertion_positions(
            len(trip.events),
            minimum_pickup_position=0,
        ):
            events = _insert_events(
                trip.events,
                request,
                pickup_position,
                dropoff_position,
            )
            evaluation = evaluate_trip(events, context)
            if not evaluation.feasible:
                continue
            provisional = _clone_trip(trip, events=events, evaluation=evaluation)
            candidate_trips = [
                provisional if existing.trip_id == trip.trip_id else existing
                for existing in state.trips.values()
            ]
            schedulable, scheduled, _ = schedule_prebooking_trips(
                candidate_trips,
                context,
            )
            if not schedulable:
                continue
            replacement = next(
                item for item in scheduled if item.trip_id == trip.trip_id
            )
            candidate = _candidate_from_replacement(
                state=state,
                old_trip=trip,
                replacement_trip=replacement,
                request=request,
                pickup_position=pickup_position,
                dropoff_position=dropoff_position,
                target_type="future_trip",
                context=context,
            )
            if candidate is not None and (
                best is None or candidate.ranking < best.ranking
            ):
                best = candidate
    return best


def _new_prebooking_candidate(
    state: DrtSystemState,
    request: TripRequest,
    context: DrtContext,
    cluster_id: int,
) -> DrtInsertionCandidate | None:
    evaluation = evaluate_trip(
        [DrtEvent(request, "pickup"), DrtEvent(request, "dropoff")],
        context,
    )
    if not evaluation.feasible:
        return None
    trip_id = state.next_trip_id
    provisional = DrtTrip(
        trip_id=trip_id,
        vehicle_id=None,
        status="planned",
        events=[DrtEvent(request, "pickup"), DrtEvent(request, "dropoff")],
        evaluation=evaluation,
        cluster_id=cluster_id,
    )
    schedulable, scheduled, _ = schedule_prebooking_trips(
        [*state.trips.values(), provisional],
        context,
    )
    if not schedulable:
        return None
    replacement = next(item for item in scheduled if item.trip_id == trip_id)
    return _candidate_from_replacement(
        state=state,
        old_trip=None,
        replacement_trip=replacement,
        request=request,
        pickup_position=0,
        dropoff_position=1,
        target_type="new_trip",
        context=context,
    )


def commit_candidate(
    state: DrtSystemState,
    candidate: DrtInsertionCandidate,
) -> None:
    if candidate.target_type == "new_trip":
        if candidate.trip_id != state.next_trip_id:
            raise ValueError("trip_id_allocation_mismatch")
        state.next_trip_id += 1
    state.trips[candidate.trip_id] = candidate.replacement_trip
    state.served_request_ids.add(candidate.request_id)
    state.rejected_requests.pop(candidate.request_id, None)
    vehicle = state.vehicles[candidate.vehicle_id]
    if candidate.trip_id not in vehicle.trip_ids:
        vehicle.trip_ids.append(candidate.trip_id)
    vehicle.trip_ids.sort(
        key=lambda trip_id: (
            state.trips[trip_id].departure_time,
            state.trips[trip_id].return_time,
            trip_id,
        )
    )


def plan_prebookings(
    state: DrtSystemState,
    clusters: list[list[TripRequest]],
    context: DrtContext,
    *,
    service_policy: str,
) -> None:
    del service_policy  # policy affects final feasibility, not candidate hard constraints.
    for cluster_id, cluster in enumerate(clusters):
        for request in sorted(
            cluster,
            key=lambda item: (item.departure_time, item.request_id),
        ):
            same_cluster_ids = [
                trip.trip_id
                for trip in state.trips.values()
                if trip.cluster_id == cluster_id and trip.status == "planned"
            ]
            candidate = find_best_prebooking_insertion(
                state,
                request,
                same_cluster_ids,
                context,
            )
            if candidate is None:
                other_ids = [
                    trip.trip_id
                    for trip in state.trips.values()
                    if trip.trip_id not in same_cluster_ids
                    and trip.status == "planned"
                    and trip.created_by == "prebooking"
                ]
                candidate = find_best_prebooking_insertion(
                    state,
                    request,
                    other_ids,
                    context,
                )
            if candidate is None:
                candidate = _new_prebooking_candidate(
                    state,
                    request,
                    context,
                    cluster_id,
                )
            if candidate is None:
                candidate = find_best_prebooking_insertion(
                    state,
                    request,
                    other_ids,
                    context,
                )
                state.record_rejection(request.request_id, "prebooking_infeasible")
                continue
            commit_candidate(state, candidate)
            if not _reschedule_prebookings(state, context):
                raise RuntimeError("committed prebooking schedule became infeasible")


def _announcement_time(request: TripRequest) -> float:
    for name in ("announcement_time", "arrival_time"):
        value = getattr(request, name, None)
        if value is not None:
            return float(value)
    return float(request.departure_time)


def build_realtime_batches(
    requests: list[TripRequest],
    context: DrtContext,
) -> list[tuple[float, list[TripRequest]]]:
    batches: defaultdict[float, list[TripRequest]] = defaultdict(list)
    interval = context.policy.recalculation_interval
    if interval is not None and interval <= 0:
        raise ValueError("recalculation_interval must be positive")
    for request in requests:
        announcement = _announcement_time(request)
        if announcement > (
            context.policy.departure_window_end + context.policy.tolerance
        ):
            batches[announcement].append(request)
            continue
        epoch = (
            announcement
            if interval is None
            else math.ceil(announcement / interval) * interval
        )
        batches[min(epoch, context.policy.departure_window_end)].append(request)
    return [
        (
            epoch,
            sorted(
                batch,
                key=lambda request: (
                    request.departure_time,
                    request.request_id,
                ),
            ),
        )
        for epoch, batch in sorted(batches.items())
    ]


def advance_system_to_epoch(
    state: DrtSystemState,
    epoch: float,
    context: DrtContext,
) -> None:
    state.current_time = float(epoch)
    for vehicle in state.vehicles.values():
        vehicle.current_time = float(epoch)
        vehicle.current_location = context.hub
        vehicle.active_trip_id = None
    for trip in state.trips.values():
        if epoch < trip.departure_time:
            trip.status = "planned"
            trip.locked_event_count = 0
        elif epoch < trip.return_time:
            trip.status = "active"
            completed = sum(
                record.event_time <= epoch + context.policy.tolerance
                for record in trip.evaluation.event_records
            )
            trip.locked_event_count = min(len(trip.events), completed + 1)
            assert trip.vehicle_id is not None
            vehicle = state.vehicles[trip.vehicle_id]
            vehicle.active_trip_id = trip.trip_id
            completed_records = [
                record
                for record in trip.evaluation.event_records
                if record.event_time <= epoch + context.policy.tolerance
            ]
            if completed_records:
                vehicle.current_location = completed_records[-1].location
        else:
            trip.status = "completed"
            trip.locked_event_count = len(trip.events)


def _vehicle_neighbors(
    state: DrtSystemState,
    trip: DrtTrip,
) -> tuple[DrtTrip | None, DrtTrip | None]:
    assert trip.vehicle_id is not None
    trips = state.vehicle_trips(trip.vehicle_id)
    index = next(
        index for index, item in enumerate(trips) if item.trip_id == trip.trip_id
    )
    previous = trips[index - 1] if index else None
    following = trips[index + 1] if index + 1 < len(trips) else None
    return previous, following


def _fits_vehicle_neighbors(
    state: DrtSystemState,
    replacement: DrtTrip,
    context: DrtContext,
) -> bool:
    old = state.trips.get(replacement.trip_id)
    if old is None:
        return True
    previous, following = _vehicle_neighbors(state, old)
    tolerance = context.policy.tolerance
    if previous is not None and replacement.departure_time < (
        previous.return_time - tolerance
    ):
        return False
    if following is not None and replacement.return_time > (
        following.departure_time + tolerance
    ):
        return False
    return True


def evaluate_trip_insertion_candidate(
    *,
    state: DrtSystemState,
    trip: DrtTrip,
    request: TripRequest,
    pickup_position: int,
    dropoff_position: int,
    epoch: float,
    context: DrtContext,
) -> DrtInsertionCandidate | None:
    if trip.vehicle_id is None:
        return None
    events = _insert_events(
        trip.events,
        request,
        pickup_position,
        dropoff_position,
    )
    previous, _ = _vehicle_neighbors(state, trip)
    if trip.status == "active":
        evaluation = evaluate_trip(
            events,
            context,
            fixed_departure_time=trip.departure_time,
        )
        target_type: CandidateTarget = "active_trip"
    else:
        earliest = max(
            float(epoch),
            0.0 if previous is None else previous.return_time,
        )
        evaluation = evaluate_trip(
            events,
            context,
            earliest_departure_time=earliest,
        )
        target_type = "future_trip"
    if not evaluation.feasible:
        return None
    replacement = _clone_trip(
        trip,
        events=events,
        evaluation=evaluation,
    )
    if not _fits_vehicle_neighbors(state, replacement, context):
        return None
    return _candidate_from_replacement(
        state=state,
        old_trip=trip,
        replacement_trip=replacement,
        request=request,
        pickup_position=pickup_position,
        dropoff_position=dropoff_position,
        target_type=target_type,
        context=context,
    )


def vehicle_free_intervals(
    state: DrtSystemState,
    vehicle_id: int,
    epoch: float,
    context: DrtContext,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    cursor = max(0.0, float(epoch))
    for trip in state.vehicle_trips(vehicle_id):
        if trip.return_time <= cursor + context.policy.tolerance:
            continue
        if trip.departure_time > cursor + context.policy.tolerance:
            intervals.append((cursor, trip.departure_time))
        cursor = max(cursor, trip.return_time)
    if cursor <= context.policy.departure_window_end + context.policy.tolerance:
        intervals.append((cursor, context.policy.latest_return_time))
    return intervals


def evaluate_new_trip_candidate(
    *,
    state: DrtSystemState,
    vehicle_id: int,
    free_interval: tuple[float, float],
    request: TripRequest,
    context: DrtContext,
) -> DrtInsertionCandidate | None:
    interval_start, interval_end = free_interval
    events = [DrtEvent(request, "pickup"), DrtEvent(request, "dropoff")]
    evaluation = evaluate_trip(
        events,
        context,
        earliest_departure_time=interval_start,
    )
    if (
        not evaluation.feasible
        or evaluation.return_time
        > min(context.policy.latest_return_time, interval_end)
        + context.policy.tolerance
    ):
        return None
    trip = DrtTrip(
        trip_id=state.next_trip_id,
        vehicle_id=vehicle_id,
        status="planned",
        events=events,
        evaluation=evaluation,
        created_by="realtime",
    )
    return _candidate_from_replacement(
        state=state,
        old_trip=None,
        replacement_trip=trip,
        request=request,
        pickup_position=0,
        dropoff_position=1,
        target_type="new_trip",
        context=context,
    )


def iter_realtime_targets(
    state: DrtSystemState,
    request: TripRequest,
    epoch: float,
    context: DrtContext,
) -> Iterator[DrtInsertionCandidate]:
    del request
    for trip in state.ordered_trips():
        if trip.status not in {"active", "planned"}:
            continue
        minimum = trip.locked_event_count if trip.status == "active" else 0
        if minimum >= len(trip.events):
            continue
        for pickup_position, dropoff_position in iter_insertion_positions(
            len(trip.events),
            minimum_pickup_position=minimum,
        ):
            yield trip, pickup_position, dropoff_position
    for vehicle_id in state.vehicles:
        for interval in vehicle_free_intervals(
            state,
            vehicle_id,
            epoch,
            context,
        ):
            yield vehicle_id, interval


def insert_realtime_request(
    state: DrtSystemState,
    request: TripRequest,
    epoch: float,
    context: DrtContext,
) -> bool:
    if _announcement_time(request) > (
        context.policy.departure_window_end + context.policy.tolerance
    ):
        state.record_rejection(
            request.request_id,
            "request_after_departure_window",
        )
        return False
    best: DrtInsertionCandidate | None = None
    for trip in state.ordered_trips():
        if trip.status not in {"active", "planned"}:
            continue
        minimum = trip.locked_event_count if trip.status == "active" else 0
        if minimum >= len(trip.events):
            continue
        for pickup_position, dropoff_position in iter_insertion_positions(
            len(trip.events),
            minimum_pickup_position=minimum,
        ):
            candidate = evaluate_trip_insertion_candidate(
                state=state,
                trip=trip,
                request=request,
                pickup_position=pickup_position,
                dropoff_position=dropoff_position,
                epoch=epoch,
                context=context,
            )
            if candidate is not None and (
                best is None or candidate.ranking < best.ranking
            ):
                best = candidate
    for vehicle_id in state.vehicles:
        for interval in vehicle_free_intervals(
            state,
            vehicle_id,
            epoch,
            context,
        ):
            candidate = evaluate_new_trip_candidate(
                state=state,
                vehicle_id=vehicle_id,
                free_interval=interval,
                request=request,
                context=context,
            )
            if candidate is not None and (
                best is None or candidate.ranking < best.ranking
            ):
                best = candidate
    if best is None:
        state.record_rejection(request.request_id, "no_feasible_insertion")
        return False
    commit_candidate(state, best)
    advance_system_to_epoch(state, epoch, context)
    return True


def _max_concurrent_trips(trips: Sequence[DrtTrip]) -> int:
    timeline: list[tuple[float, int]] = []
    for trip in trips:
        timeline.append((trip.departure_time, 1))
        timeline.append((trip.return_time, -1))
    concurrent = 0
    maximum = 0
    for _, delta in sorted(timeline, key=lambda item: (item[0], item[1])):
        concurrent += delta
        maximum = max(maximum, concurrent)
    return maximum


def validate_system(
    state: DrtSystemState,
    context: DrtContext,
) -> tuple[bool, tuple[str, ...]]:
    violations: list[str] = []
    request_counts: Counter[int] = Counter()
    for trip in state.trips.values():
        request_counts.update(trip.request_ids)
        if not trip.evaluation.feasible:
            violations.extend(trip.evaluation.violations)
        if trip.vehicle_id not in state.vehicles:
            violations.append("vehicle_unavailable")
        if trip.departure_time > (
            context.policy.departure_window_end + context.policy.tolerance
        ):
            violations.append("departure_window")
        if trip.return_time > (
            context.policy.latest_return_time + context.policy.tolerance
        ):
            violations.append("overtime_limit")
        expected_status: TripStatus
        if state.current_time < trip.departure_time:
            expected_status = "planned"
        elif state.current_time < trip.return_time:
            expected_status = "active"
        else:
            expected_status = "completed"
        if trip.status != expected_status:
            violations.append("trip_status_mismatch")
    if set(request_counts) != state.served_request_ids or any(
        count != 1 for count in request_counts.values()
    ):
        violations.append("duplicate_or_missing_served_request")
    for vehicle_id in state.vehicles:
        trips = state.vehicle_trips(vehicle_id)
        for previous, following in zip(trips, trips[1:]):
            if following.departure_time < (
                previous.return_time - context.policy.tolerance
            ):
                violations.append("vehicle_schedule_conflict")
    total_distance, operating_time, total_trips, served, expenditure = _current_totals(
        state,
        context,
    )
    del total_distance, operating_time, total_trips, served
    if expenditure > (
        context.policy.benchmark_expenditure + context.policy.tolerance
    ):
        violations.append("expenditure_limit")
    unique = tuple(dict.fromkeys(violations))
    return not unique, unique


def summarize_system(
    state: DrtSystemState,
    context: DrtContext,
) -> DrtPlanSummary:
    trips = [trip for trip in state.trips.values() if trip.events]
    total_wait = sum(trip.evaluation.total_wait for trip in trips)
    total_onboard = sum(trip.evaluation.total_onboard for trip in trips)
    total_distance = sum(trip.travel_distance for trip in trips)
    operating_time = sum(trip.operating_time for trip in trips)
    total_trips = len(trips)
    served = len(state.served_request_ids)
    expenditure = context.expenditure_fn(
        total_distance,
        total_trips,
        served,
        operating_time,
    )
    valid, violations = validate_system(state, context)
    return DrtPlanSummary(
        served_requests=served,
        rejected_requests=len(state.rejected_requests),
        total_wait=total_wait,
        total_onboard=total_onboard,
        total_travel_distance=total_distance,
        operating_time=operating_time,
        total_trips=total_trips,
        max_concurrent_trips=_max_concurrent_trips(trips),
        vehicle_reuse_ratio=(
            None
            if context.vehicle_count <= 0
            else total_trips / context.vehicle_count
        ),
        net_expenditure=expenditure,
        constraints_satisfied=valid,
        violations=violations,
    )

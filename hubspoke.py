



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


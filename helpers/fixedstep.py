import math


import networkx as nx
import helpers.fleet_sizing as fleet_sizing
import helpers.functions as fs
from helpers.config import LoopContext
from helpers.types import GridNode
# from helpers.netx import NetworkContext

#grid network loops
#TODO
def build_context(network_context, config, nets, fleet): 
    graph = network_context.graph
    routes = network_context.routes
    route_vehicle_ids = _distr_route_vehicle_ids(len(routes), fleet.num)
    loops = tuple(
        _build_loop_sub(nets, route, vehicle_ids, graph)
        for route, vehicle_ids in zip(routes, route_vehicle_ids)
        if vehicle_ids
    )
    if not loops:
        raise ValueError("fleet must contain at least one vehicle for route service")
    # fs.draw_loops(loops, Path(__file__).resolve().parent / "rs" / "loops")

    weighted_contexts = {loop.id: _loop_context(loop, graph) for loop in loops} 
    headway = float(fleet.freq)
    span = float(config.span)
    departure_count = int(math.ceil(span / headway)) + 1

    total_trips = 0
    total_travel_distance = 0.0
    infeasible_routes: list[str] = []

    for loop in loops:
        route_length = float(weighted_contexts[loop.id]["length"])
        assigned_vehicles = len(loop.vehicle_offsets)
        required_vehicles = fleet_sizing.required_grid_route_vehicle_count(
            nets,
            loop.length,
            fleet,
        )
        if assigned_vehicles < required_vehicles:
            infeasible_routes.append(loop.id)
        total_trips += departure_count
        total_travel_distance += float(departure_count) * route_length
    metrics = {
        "feasible": not infeasible_routes, # not [] == true
        "feasibility_reason": (
            "feasible"
            if not infeasible_routes
            else "insufficient_vehicles_for_headway"
        ),
        "infeasible_routes": tuple(infeasible_routes),
        "total_trips": float(total_trips),
        "total_travel_distance": float(total_travel_distance),
    }
    return loops, weighted_contexts, metrics
# loops: 包含路线节点、固定站点索引、车辆分配 

# 将车辆分配给各条路线
def _distr_route_vehicle_ids(route_count: int, vehicle_count: int) -> tuple[tuple[int, ...], ...]: # 将车辆分配给各条路线
    base_count, remainder = divmod(vehicle_count, route_count)
    route_vehicle_ids = []
    next_vehicle_id = 0
    for route_index in range(route_count):
        count = base_count + int(route_index < remainder)
        ids = tuple(range(next_vehicle_id, next_vehicle_id + count))
        route_vehicle_ids.append(ids)
        next_vehicle_id += count
    return tuple(route_vehicle_ids)



# 构建单条路线的LoopContext
def _build_loop_sub(nets, route, route_vehicle_ids: tuple[int, ...], graph: nx.Graph) -> LoopContext: 

 # 将路线的停靠点扩展为完整的节点序列，沿着图中的最短路径连接每对连续停靠点
    ordered_stops = route.stops
    nodes: list[GridNode] = []
    for index, stop in enumerate(ordered_stops):
        next_stop = ordered_stops[(index + 1) % len(ordered_stops)]
        segment = nx.shortest_path(graph, stop, next_stop, weight="weight")
        if not nodes:
            nodes.extend(segment)
        else:
            nodes.extend(segment[1:])
    
    route_positions = {node: index for index, node in enumerate(nodes[:-1])}
    optional_anchor_indices: dict[GridNode, int] = {}
    route_set = set(nodes[:-1])

    for node in nodes[:-1]:
        anchor_index = route_positions[node]
        if getattr(nets, "_type", None) == "grid" and _is_grid_node(node):
            x_coord, y_coord = node
            max_dev = max(0, int(float(getattr(nets, "max_dev", 0))))
            for delta in range(1, max_dev + 1):
                for candidate in ((x_coord, y_coord + delta), (x_coord, y_coord - delta)):
                    if candidate in route_set or candidate not in graph:
                        continue
                    current_index = optional_anchor_indices.get(candidate)
                    if current_index is None or anchor_index < current_index:
                        optional_anchor_indices[candidate] = anchor_index
        else:
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

# edge权重
def _loop_context(
    loop: LoopContext,
    graph: nx.Graph,
) -> dict:
    offsets = {} # dict[GridNode, float]
    cumulative = 0.0
    for index, node in enumerate(loop.nodes[:-1]):  # 计算循环路线中每个节点的offset（相对于路线起点的距离），基于图中的边权重累积计算
        offsets[node] = cumulative
        #获取图中两个节点之间的边权重，如果没有指定权重则默认为1.0
        cumulative += float(graph[node][loop.nodes[index + 1]].get("weight", 1.0))
    return {
        "offsets": offsets,
        "length": float(cumulative),
    }
def _node_sort_key(node: GridNode) -> tuple[str, str]:
    return (type(node).__name__, repr(node))


def _is_grid_node(node: GridNode) -> bool:
    return isinstance(node, tuple) and len(node) == 2

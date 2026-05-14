from __future__ import annotations
import json
import math
from dataclasses import dataclass, replace
from typing import Any

import networkx as nx
import numpy as np
from pathlib import Path
import pandas as pd

NetworkNode = Any # type alias: network node（网络节点）


@dataclass(frozen=True, slots=True)
class TripRequest:
    request_id: int
    origin: NetworkNode
    destination: NetworkNode
    departure_time: int
    request_type: str = "real_time"


def _build_grid_nodes(grid_size: int) -> list[NetworkNode]:
    return [(x, y) for x in range(grid_size) for y in range(grid_size)]


def _normalize_weights(weights: np.ndarray) -> np.ndarray: # 归一化权重
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(weights.shape, 1.0 / weights.size, dtype=float)
    return weights / total


def _hotspot_weights(nodes: list[NetworkNode], hotspot: NetworkNode) -> np.ndarray: # 计算每个节点到热点的曼哈顿距离，并将距离转换为权重，距离越近权重越大
    distances = np.array(
        [abs(node[0] - hotspot[0]) + abs(node[1] - hotspot[1]) for node in nodes],
        dtype=float,
    )
    return np.exp(-0.3 * distances)


def _network_hotspot_weights(
    graph: nx.Graph,
    nodes: list[NetworkNode],
    hotspot: NetworkNode,
) -> np.ndarray:
    if hotspot in graph:
        lengths = nx.single_source_dijkstra_path_length(
            graph,
            hotspot,
            weight="weight",
        )
        distances = np.array([float(lengths[node]) for node in nodes], dtype=float)
        return np.exp(-0.3 * distances)

    positions = nx.get_node_attributes(graph, "pos")
    if (
        isinstance(hotspot, (tuple, list))
        and len(hotspot) == 2
        and all(isinstance(value, (int, float)) for value in hotspot)
        and all(node in positions for node in nodes)
    ):
        hx, hy = float(hotspot[0]), float(hotspot[1])
        distances = np.array(
            [
                math.hypot(float(positions[node][0]) - hx, float(positions[node][1]) - hy)
                for node in nodes
            ],
            dtype=float,
        )
        return np.exp(-0.3 * distances)

    return np.full(len(nodes), 1.0 / len(nodes), dtype=float)


def _mix_spatial_weights( # 混合空间权重
    uniform_weights: np.ndarray,
    hotspot_weights: np.ndarray,
    heterogeneity: float,
) -> np.ndarray:
    clipped = float(np.clip(heterogeneity, 0.0, 1.0))
    mixed = (1.0 - clipped) * uniform_weights + clipped * hotspot_weights
    return _normalize_weights(mixed)


def _build_temporal_weights(config, heterogeneity: float) -> np.ndarray:
    clipped = float(np.clip(heterogeneity, 0.0, 1.0))
    minutes = np.arange(config.span, dtype=float)
    uniform = np.full(config.span, 1.0 / config.span, dtype=float)
    peak_weights = [
        np.exp(-0.5 * ((minutes - peak) / 10.0) ** 2)
        for peak in config.peaks
    ]
    peaked = _normalize_weights(np.sum(peak_weights, axis=0))
    mixed = (1.0 - clipped) * uniform + clipped * peaked
    return _normalize_weights(mixed)


def generate_requests(
    config,
    nets,
    scenario: dict,
    network_context=None,
) -> list[TripRequest]:
    rng = np.random.default_rng(int(scenario["seed"]))
    request_count = int(rng.poisson(lam=float(scenario["lambda"])))
    if request_count <= 0:
        return []

    if network_context is None or network_context.network_type == "grid":
        nodes = _build_grid_nodes(nets.grid)
        origin_hotspot_weights = _hotspot_weights(nodes, config.o_hotspot)
        destination_hotspot_weights = _hotspot_weights(nodes, config.d_hotspot)
    else:
        nodes = list(network_context.request_nodes)
        origin_hotspot_weights = _network_hotspot_weights(
            network_context.graph,
            nodes,
            config.o_hotspot,
        )
        destination_hotspot_weights = _network_hotspot_weights(
            network_context.graph,
            nodes,
            config.d_hotspot,
        )

    node_indices = np.arange(len(nodes))
    minute_indices = np.arange(config.span)
    uniform_node_weights = np.full(len(nodes), 1.0 / len(nodes), dtype=float)

    origin_weights = _mix_spatial_weights(
        uniform_node_weights,
        origin_hotspot_weights,
        float(scenario["hs"]),
    )
    destination_weights = _mix_spatial_weights(
        uniform_node_weights,
        destination_hotspot_weights,
        float(scenario["hs"]),
    )
    temporal_weights = _build_temporal_weights(config, float(scenario["ht"]))

    requests: list[TripRequest] = []
    for request_id in range(request_count):
        origin_index = int(rng.choice(node_indices, p=origin_weights))
        destination_index = int(rng.choice(node_indices, p=destination_weights))
        while destination_index == origin_index:
            destination_index = int(rng.choice(node_indices, p=destination_weights))

        departure_time = int(rng.choice(minute_indices, p=temporal_weights))
        requests.append(
            TripRequest(
                request_id=request_id,
                origin=nodes[origin_index],
                destination=nodes[destination_index],
                departure_time=departure_time,
            )
        )

    return sorted(requests, key=lambda request: (request.departure_time, request.request_id))


def assign_request_types(
    requests: list[TripRequest],
    alpha: float,
    seed: int | None,
) -> list[TripRequest]:
    prebooking_count = int(np.floor(float(np.clip(alpha, 0.0, 1.0)) * len(requests)))
    if prebooking_count <= 0:
        return [replace(request, request_type="real_time") for request in requests]

    rng = np.random.default_rng(seed)
    prebooking_indices = set(
        int(index)
        for index in rng.choice(len(requests), size=prebooking_count, replace=False)
    )
    typed_requests = []
    for index, request in enumerate(requests):
        request_type = "pre_booking" if index in prebooking_indices else "real_time"
        typed_requests.append(replace(request, request_type=request_type))
    return typed_requests

def request_to_record(request: TripRequest) -> dict:
    return {
        "request_id": request.request_id,
        "origin": _node_to_record(request.origin),
        "destination": _node_to_record(request.destination),
        "departure_time": request.departure_time,
        "request_type": request.request_type,
    }


def _node_to_record(node: NetworkNode):
    if isinstance(node, tuple):
        return list(node)
    return node


def save_requests(
    requests: list[TripRequest],
    output_dir: Path,
    file_name: str = "requests.json",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name
    records = [request_to_record(request) for request in requests]

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    return output_path


def load_requests(path: Path) -> list[TripRequest]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    return [
        TripRequest(
            request_id=int(record["request_id"]),
            origin=_record_to_node(record["origin"]),
            destination=_record_to_node(record["destination"]),
            departure_time=int(record["departure_time"]),
            request_type=str(record.get("request_type", "real_time")),
        )
        for record in records
    ]


def _record_to_node(value):
    if isinstance(value, list):
        return tuple(value)
    return value

def avg_served(frame: pd.DataFrame, division: pd.DataFrame, ac_rate: str| None = None) -> pd.DataFrame:
    plot_frame = frame.copy()
    served_num = plot_frame["served_requests"]
    plot_frame["avg_net_expenditure"] = (division.iloc[:, 0] / served_num).fillna(0.0)
    plot_frame["avg_service_time"] = (division.iloc[:, 1] / served_num).fillna(0.0)
    if ac_rate == "acceptance":
        plot_frame["acceptance_rate"] = (served_num / plot_frame["total_requests"]).fillna(0.0)
    else: pass
    return plot_frame

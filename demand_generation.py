from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GridNode = tuple[int, int] # 定义一个type，表示网格中的一个节点（x, y）


@dataclass(frozen=True, slots=True)
class TripRequest:
    request_id: int
    origin: GridNode
    destination: GridNode
    departure_time: int


def _build_grid_nodes(grid_size: int) -> list[GridNode]:
    return [(x, y) for x in range(grid_size) for y in range(grid_size)]


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(weights.shape, 1.0 / weights.size, dtype=float)
    return weights / total


def _hotspot_weights(nodes: list[GridNode], hotspot: GridNode) -> np.ndarray:
    distances = np.array(
        [abs(node[0] - hotspot[0]) + abs(node[1] - hotspot[1]) for node in nodes],
        dtype=float,
    )
    return np.exp(-0.6 * distances)


def _mix_spatial_weights(
    uniform_weights: np.ndarray,
    hotspot_weights: np.ndarray,
    heterogeneity: float,
) -> np.ndarray:
    clipped = float(np.clip(heterogeneity, 0.0, 1.0))
    mixed = (1.0 - clipped) * uniform_weights + clipped * hotspot_weights
    return _normalize_weights(mixed)


def _build_temporal_weights(horizon: int, heterogeneity: float) -> np.ndarray:
    clipped = float(np.clip(heterogeneity, 0.0, 1.0))
    minutes = np.arange(horizon, dtype=float)
    uniform = np.full(horizon, 1.0 / horizon, dtype=float)
    peak_one = np.exp(-0.5 * ((minutes - 45.0) / 10.0) ** 2)
    peak_two = np.exp(-0.5 * ((minutes - 120.0) / 10.0) ** 2)
    peaked = _normalize_weights(peak_one + peak_two)
    mixed = (1.0 - clipped) * uniform + clipped * peaked
    return _normalize_weights(mixed)


def generate_requests(
    lambda_value: float,
    hs: float,
    ht: float,
    seed: int,
    grid_size: int = 9,
    horizon: int = 180,
) -> list[TripRequest]:
    rng = np.random.default_rng(seed)
    request_count = int(rng.poisson(lam=float(lambda_value)))
    if request_count <= 0:
        return []

    nodes = _build_grid_nodes(grid_size)
    node_indices = np.arange(len(nodes))
    minute_indices = np.arange(horizon)
    uniform_node_weights = np.full(len(nodes), 1.0 / len(nodes), dtype=float)

    origin_hotspot = (1, 1)
    destination_hotspot = (grid_size - 2, grid_size - 2)
    origin_weights = _mix_spatial_weights(
        uniform_node_weights,
        _hotspot_weights(nodes, origin_hotspot),
        hs,
    )
    destination_weights = _mix_spatial_weights(
        uniform_node_weights,
        _hotspot_weights(nodes, destination_hotspot),
        hs,
    )
    temporal_weights = _build_temporal_weights(horizon, ht)

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

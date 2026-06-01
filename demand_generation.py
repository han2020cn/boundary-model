from __future__ import annotations
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any
import networkx as nx
import numpy as np
from pathlib import Path
import pandas as pd

import netx as net
import plt_draw as plt
from config import TripRequest
NetworkNode = Any # type alias: network node（网络节点）


def _weights_normalize(weights: np.ndarray) -> np.ndarray: # 归一化权重
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(weights.shape, 1.0 / weights.size, dtype=float)
    return weights / total


def _weights_hotspot(nodes: list[NetworkNode], hotspot: NetworkNode) -> np.ndarray: # 计算每个节点到热点的曼哈顿距离，并将距离转换为权重，距离越近权重越大
    distances = np.array(
        [abs(node[0] - hotspot[0]) + abs(node[1] - hotspot[1]) for node in nodes],
        dtype=float,
    )
    return np.exp(-0.3 * distances)


def _weights_network_hotspot(
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
        return np.exp(-0.3 * 0.1 * distances) # 将距离转换为权重，距离越近权重越大

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


def _weights_mix_spatial( # 混合空间权重
    uniform_weights: np.ndarray,
    hotspot_weights: np.ndarray,
    heterogeneity: float,
) -> np.ndarray:

    mixed = (1.0 - heterogeneity) * uniform_weights + heterogeneity * hotspot_weights
    return _weights_normalize(mixed)


def _weights_build_temporal(config, heterogeneity: float) -> np.ndarray:
    clipped = float(np.clip(heterogeneity, 0.0, 1.0))
    minutes = np.arange(config.span, dtype=float)
    uniform = np.full(config.span, 1.0 / config.span, dtype=float)
    peak_width = float(getattr(config, "peak_width_minutes", 30))
    peak_weights = [
        np.exp(-0.5 * ((minutes - peak) / peak_width) ** 2)
        for peak in config.peaks
    ]
    peaked = _weights_normalize(np.sum(peak_weights, axis=0))
    mixed = (1.0 - clipped) * uniform + clipped * peaked
    return _weights_normalize(mixed)


def _requests_generate(
    config,
    nets,
    scenario: dict,
    network_context=None,
) -> list[TripRequest]:
    rng = np.random.default_rng(int(scenario["seed"]))
    lambda_per_hour = float(scenario["lambda"]) # 每小时需求强度（demand intensity per hour）
    mean_count = lambda_per_hour * config.span / 60.0       # 计算在给定时间范围内的平均请求数量（mean request count over the time horizon）
    request_count = int(rng.poisson(lam=mean_count))
    if request_count <= 0:
        raise ValueError("No request generated")

    nodes = list(network_context.request_nodes)
    if len(nodes) < 2:
        raise ValueError("request generation requires at least two request nodes")

    #目前的空间衰减exp(-0.3 * 0.1 * distances)
    origin_hotspot_weights = _weights_network_hotspot(
        network_context.graph,
        nodes,
        config.o_hotspot,
    )
    destination_hotspot_weights = _weights_network_hotspot(
        network_context.graph,
        nodes,
        config.d_hotspot,
    )

    node_indices = np.arange(len(nodes))
    minute_indices = np.arange(config.span)
    uniform_node_weights = np.full(len(nodes), 1.0 / len(nodes), dtype=float)

    origin_weights = _weights_mix_spatial(
        uniform_node_weights,
        origin_hotspot_weights,
        float(scenario["hs"]),
    )
    destination_weights = _weights_mix_spatial(
        uniform_node_weights,
        destination_hotspot_weights,
        float(scenario["hs"]),
    )
    temporal_weights = _weights_build_temporal(config, float(scenario["ht"]))

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
    print(f"Generated {len(requests)} requests with lambda={lambda_per_hour}, hs={scenario['hs']}, ht={scenario['ht']}")
    plt._draw_request(requests, nets, Path(__file__).resolve().parent / "rs"/"requests")
    return sorted(requests, key=lambda request: (request.departure_time, request.request_id))


def _request_types_assign(
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




def _requests_save(
    requests: list[TripRequest],
    output_dir: Path,
    date: str,
) -> Path:
    file_name = f"requests_{date}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "requests" / file_name
    records = [
        {
            "request_id": request.request_id,
            "origin": request.origin,
            "destination": request.destination,
            "departure_time": request.departure_time,
            "request_type": request.request_type,
        }
        for request in requests
    ]

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


def _record_to_node(value): # 将记录中的节点表示转换回网络节点对象，如果是列表则转换为元组，否则保持原样
    if isinstance(value, list):
        return tuple(value)
    return value

def avg_served(frame: pd.DataFrame, division: pd.DataFrame, ac_rate: str| None = None) -> pd.DataFrame:
    # 计算平均服务指标：平均净支出、平均服务时间，以及如果指定了接受率计算，则计算接受率
    plot_frame = frame.copy()
    served_num = plot_frame["served_requests"]
    plot_frame["avg_net_expenditure"] = (division.iloc[:, 0] / served_num).fillna(0.0)
    plot_frame["avg_service_time"] = (division.iloc[:, 1] / served_num).fillna(0.0)
    if ac_rate == "acceptance":
        plot_frame["acceptance_rate"] = (served_num / plot_frame["total_requests"]).fillna(0.0)
    else: pass
    return plot_frame

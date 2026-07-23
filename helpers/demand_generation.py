from __future__ import annotations
import json
import math
from numbers import Real
from datetime import datetime
import networkx as nx
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import helpers.netx as net
import plt_draw
import helpers.functions as fs
from helpers.config import HotspotConfig, TripRequest
from helpers.types import NetworkNode


def _weights_normalize(weights: np.ndarray) -> np.ndarray: # 归一化权重
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(weights.shape, 1.0 / weights.size, dtype=float)
    return weights / total


def _is_hotspot_coordinate(value) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(
            isinstance(coordinate, Real)
            and not isinstance(coordinate, bool)
            and math.isfinite(float(coordinate))
            for coordinate in value
        )
    )


def _normalize_hotspots(hotspot: HotspotConfig) -> tuple[tuple[float, float], ...]:
    if _is_hotspot_coordinate(hotspot):
        return ((float(hotspot[0]), float(hotspot[1])),)

    if not isinstance(hotspot, (tuple, list)) or not hotspot:
        raise ValueError(
            "hotspot must be a coordinate (x, y) or a non-empty sequence "
            "of coordinates"
        )

    normalized = []
    for index, coordinate in enumerate(hotspot):
        if not _is_hotspot_coordinate(coordinate):
            raise ValueError(
                f"hotspot[{index}] must contain exactly two finite numbers"
            )
        normalized.append((float(coordinate[0]), float(coordinate[1])))
    return tuple(normalized)


def _weights_network_hotspot(
    graph: nx.Graph,
    nodes: list[NetworkNode],
    hotspot: HotspotConfig,
) -> np.ndarray:
    hotspots = _normalize_hotspots(hotspot)
    positions = nx.get_node_attributes(graph, "pos")
    invalid_position_nodes = [
        node
        for node in nodes
        if node not in positions or not _is_hotspot_coordinate(positions[node])
    ]
    if invalid_position_nodes:
        preview = ", ".join(repr(node) for node in invalid_position_nodes[:5])
        if len(invalid_position_nodes) > 5:
            preview += f", ... {len(invalid_position_nodes) - 5} more"
        raise ValueError(f"request nodes require finite two-dimensional pos: {preview}")

    hotspot_distributions = []
    for hx, hy in hotspots:
        distances = np.array(
            [
                math.hypot(float(positions[node][0]) - hx, float(positions[node][1]) - hy)
                for node in nodes
            ],
            dtype=float,
        )
        # Each hotspot contributes the same total probability mass.
        hotspot_distributions.append(
            _weights_normalize(np.exp(-0.3 * distances))
        )
    return _weights_normalize(
        np.mean(np.stack(hotspot_distributions, axis=0), axis=0)
    )


def _weights_mix_spatial( # 混合空间权重
    uniform_weights: np.ndarray,
    hotspot_weights: np.ndarray,
    heterogeneity: float,
) -> np.ndarray:

    hotspot_distribution = _weights_normalize(hotspot_weights)
    mixed = (
        (1.0 - heterogeneity) * uniform_weights
        + heterogeneity * hotspot_distribution
    )
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


def requests_generate(
    config,
    nets,
    scenario: dict,
    network_context=None,
    *,
    fixed_seed: bool,
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

    #目前的空间衰减exp(coefficient* distances)
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
    minute_indices = np.arange(config.span)  # 减去30分钟，确保请求在时间范围内有足够的时间完成
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
    # 预订请求占比由prebooking_alpha控制
    prebooking_count = int(config.pre_alpha * request_count)
    prebooking_indices = set(
        int(index)
        for index in rng.choice(request_count, size=prebooking_count, replace=False)
    )
    for request_id in range(request_count):
        origin_index = int(rng.choice(node_indices, p=origin_weights))
        destination_index = int(rng.choice(node_indices, p=destination_weights))
        while destination_index == origin_index:
            destination_index = int(rng.choice(node_indices, p=destination_weights))

        departure_time = int(rng.choice(minute_indices, p=temporal_weights))
        request_type = 1 if request_id in prebooking_indices else 0
        requests.append(
            TripRequest(
                request_id=request_id,
                origin=nodes[origin_index],
                destination=nodes[destination_index],
                departure_time=departure_time,
                request_type = request_type,
            )
        )

    print(f"Generated {len(requests)} requests with lambda={lambda_per_hour}, hs={scenario['hs']}, ht={scenario['ht']}")
    return sorted(requests, key=lambda request: (request.departure_time, request.request_id))


def _requests_save(
    requests: list[TripRequest],
    output_dir: Path,
    date: str,
) -> Path:
    file_name = f"requests_{date}.json"
    output_dir.mkdir(parents=True, exist_ok=True)   #确保输出目录存在
    output_path = output_dir / file_name
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

    return fs.write_json(records, output_path)


def load_requests(path: Path) -> list[TripRequest]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    return [
        TripRequest(
            request_id=int(record["request_id"]),
            origin=_record_to_node(record["origin"]),
            destination=_record_to_node(record["destination"]),
            departure_time=int(record["departure_time"]),
            request_type=int(record["request_type"]),
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
    plot_frame["avg_wait"] = (division.iloc[:, 2] / served_num).fillna(0.0)
    plot_frame["avg_walk"] = (division.iloc[:, 3] / served_num).fillna(0.0)
    plot_frame["avg_onboard"] = (division.iloc[:, 4] / served_num).fillna(0.0)
    if ac_rate == "acceptance":
        plot_frame["acceptance_rate"] = (served_num / plot_frame["total_requests"]).fillna(0.0)
    else: pass
    return plot_frame


# Draw 请求起终点 didtribution
def draw_distribution(
    input_file: str|list[TripRequest],
    nets,
    scenario,
    output_dir: Path | None = None,
    date: str | None = None,
) -> Path:
    png_name = f"lambda{scenario['lambda']}_ht{scenario['ht']}_hs{scenario['hs']}_{date}.png" #png文件名
    if isinstance(input_file, list):
        requests = input_file
    else:
        requests = load_requests(Path(output_dir/input_file))
    context = net.build_network_context(nets)
    graph = context.graph
    network_type = context.network_type
    _validate_nodes(requests, graph)

    if network_type == "grid":
        pos = _grid_node_positions(graph, nets)
    else:
        raise ValueError("network_type must be 'grid' or 'hub_spoke'")

    output_path = Path(output_dir) / png_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False, sharey=False)

    # _draw_network_base(axes[0], graph, pos, context.hub, labels=None)
    _draw_request_points(axes, requests, pos)
    axes[0].set_title("origins")
    axes[1].set_title("destinations")
    for ax in axes:
        ax.set_xlim(-10 , nets.grid )
        ax.set_ylim(-10 , nets.grid )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Request Origins and Destinations ({network_type})")
        ax.set_aspect("auto", adjustable="box")
        ax.legend(loc="best", framealpha=0.88)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _draw_timeline(
    requests: list[TripRequest],
    span: int,
    scenario: dict,
    output_dir: Path,
    date: str,
) -> Path:
    """Draw the number of departing requests in each unit of the time span."""
    if not isinstance(span, (int, np.integer)) or isinstance(span, bool) or span <= 0:
        raise ValueError("span must be a positive integer")

    departure_times = np.asarray(
        [request.departure_time for request in requests],
        dtype=int,
    )
    if departure_times.size and (
        np.any(departure_times < 0) or np.any(departure_times >= span)
    ):
        raise ValueError("request departure_time must be within [0, span)")

    counts = np.bincount(departure_times, minlength=span)
    unit_times = np.arange(span)
    output_path = Path(output_dir) / (
        f"timeline_lambda{scenario['lambda']}_ht{scenario['ht']}_"
        f"hs{scenario['hs']}_{date}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(unit_times, counts, width=1.0, align="edge")
    ax.set_xlim(0, span)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Departure time")
    ax.set_ylabel("Requests per unit time")
    ax.set_title(
        "Request departure-time distribution "
        f"($\\lambda$={scenario['lambda']}, "
        f"$h_t$={scenario['ht']}, $h_s$={scenario['hs']})"
    )
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _validate_nodes(requests, graph) -> None:
    missing = []
    for request in requests:
        if request.origin not in graph:
            missing.append((request.request_id, "origin", request.origin))
        if request.destination not in graph:
            missing.append((request.request_id, "destination", request.destination))

    if missing:
        details = ", ".join(
            f"request {request_id} {field}={node!r}"
            for request_id, field, node in missing[:10]
        )
        if len(missing) > 10:
            details += f", ... {len(missing) - 10} more"
        raise ValueError(f"requests contain nodes outside graph: {details}")


def _grid_node_positions(graph, nets) -> dict: # 根据网格网络的节点坐标计算位置
    grid_len = float(getattr(nets, "grid_len", 1.0))
    positions = {}
    for node in graph.nodes:
        if (
            not isinstance(node, tuple)
            or len(node) != 2
            or not all(isinstance(value, (int, float)) for value in node)
        ):
            raise ValueError(f"grid node must be a numeric (x, y) tuple: {node!r}")
        positions[node] = (float(node[0]), float(node[1]) )
    return positions



def _draw_request_points(axes, requests, pos: dict) -> None:
    #  2D histogram heatmap 也可以考虑用热力图
    origin_x = np.asarray([pos[request.origin][0] for request in requests])
    origin_y = np.asarray([pos[request.origin][1] for request in requests])
    destination_x = np.asarray([pos[request.destination][0] for request in requests])
    destination_y = np.asarray([pos[request.destination][1] for request in requests])
    
    plt_draw.add_scatter(axes[0], records = None,
                x_key = origin_x,
                y_key = origin_y,
                )
    
    plt_draw.add_scatter(axes[1], records = None,
                x_key = destination_x,
                y_key = destination_y,
                )
    

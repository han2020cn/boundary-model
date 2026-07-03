from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import helpers.demand_generation as dg
import helpers.netx as net
import helpers.functions as fs


# ===== 2. 为不同 mode_id 定义颜色 =====

MODE_COLORS = {
    1: "tab:grey",
    2: "tab:blue",
    3: "tab:orange",
    4: "tab:red",
}


COST_TRADEOFF_LINESTYLES = {
    1: "-",
    2: "--",
    3: (0, (6, 2, 1, 2)),
    4: ":",
}

COST_TRADEOFF_MARKERS = ("*", "X", "^", "D", "o", "s", "P")



_DEFAULT_REGRESSION_TIMESTAMP: str | None = None

# Draw 请求起终点 on the network
def _draw_request(
    input_file: str|list[TripRequest],
    nets,
    output_dir: Path | None = None,
    date: str | None = None,
) -> Path:
    png_name = f"distribution_{date}.png" #png文件名
    if isinstance(input_file, list):
        requests = input_file
    else:
        requests = dg.load_requests(Path(output_dir/input_file))
    context = net.build_network_context(nets)
    graph = context.graph
    network_type = context.network_type
    _validate_nodes(requests, graph)

    if network_type == "grid":
        pos = _grid_node_positions(graph, nets)
    elif network_type == "hub_spoke":
        pos = _hub_spoke_node_positions(graph)
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


def _hub_spoke_node_positions(graph) -> dict:
    positions = {}
    missing_nodes = []
    invalid_nodes = []

    for node, data in graph.nodes(data=True):
        if "pos" not in data:
            missing_nodes.append(node)
            continue
        raw_pos = data["pos"]
        if (
            not isinstance(raw_pos, tuple)
            or len(raw_pos) != 2
            or not all(isinstance(value, (int, float)) for value in raw_pos)
        ):
            invalid_nodes.append(node)
            continue
        positions[node] = (float(raw_pos[0]), float(raw_pos[1]))

    if missing_nodes:
        raise ValueError(f"hub_spoke graph nodes missing 'pos': {missing_nodes[:10]}")
    if invalid_nodes:
        raise ValueError(f"hub_spoke graph nodes have invalid 'pos': {invalid_nodes[:10]}")
    return positions


def _draw_request_points(axes, requests, pos: dict) -> None:
    #  2D histogram heatmap 也可以考虑用热力图
    origin_x = np.asarray([pos[request.origin][0] for request in requests])
    origin_y = np.asarray([pos[request.origin][1] for request in requests])
    destination_x = np.asarray([pos[request.destination][0] for request in requests])
    destination_y = np.asarray([pos[request.destination][1] for request in requests])
    
    add_scatter(axes[0], records = None,
                x_key = origin_x,
                y_key = origin_y,
                )
    
    add_scatter(axes[1], records = None,
                x_key = destination_x,
                y_key = destination_y,
                )
    



#3d scatter plot
def plts_3d(frame: pd.DataFrame,
            output_dir: Path,
            date_stamp: str,
            x_key: str, y_key: str, z_key: str,
            offset: bool) -> Path:

    outpng = Path(output_dir) / f"3d_{date_stamp}.png"

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    model_3d(ax, frame, x_key, y_key, z_key, offset)
    ax.legend(
        title="mode_id",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    plt.tight_layout(rect=(0.0, 0.0, 0.72, 1.0))
    plt.savefig(outpng, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return outpng

# 2d scatter plot
def plts_2d(
    frame: pd.DataFrame,
    output_path: Path,
    date_stamp: str,
    x_key,
    y_key,
    types: list[str],
    show_ols: bool = True,
    show_ci: bool = True,
    show_lowess: bool = False,
    export_regression: bool = True,
    confidence: float = 0.95,
    regression_timestamp: str | None = None,
) -> Path:
    outpng = Path(output_path)/f"2d_plots_{date_stamp}.png"
    outpng.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    _draw_2d_scatter(ax, frame, x_key, y_key, types, "x-y scatter plot", show_legend=True)
    _draw_regression_overlays(
        ax,
        frame,
        x_key,
        y_key,
        show_ols=show_ols,
        show_ci=show_ci,
        show_lowess=show_lowess,
        confidence=confidence,
    )
    # 导出回归结果
    if export_regression: 
        report_frame = fs._regression_report(
            frame,
            output_path,
            x_key,
            y_key,
            plot_name="2d",
            panel="main",
            confidence=confidence,
        )


    plt.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    plt.savefig(outpng, dpi=600, bbox_inches="tight")
    # plt.show()
    plt.close(fig)
    return outpng


def plts_2d_pair(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    output_dir: Path,
    date_stamp,
    x_key,
    x1_key,
    y_key,
    y1_key,
    types: list[str],
    left_title: str,
    right_title: str,
    prebooking_alpha: float | None = None,
    show_ols: bool = True,
    show_ci: bool = True,
    show_lowess: bool = False,
    export_regression: bool = True,
    confidence: float = 0.95,
    regression_timestamp: str | None = None,
) -> Path:
    outpng = Path(output_dir) / f"2d_pairs_{date_stamp}.png"
    outpng.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False, sharey=False)
    _draw_2d_scatter(axes[0], left_frame, x_key, y_key, types, left_title, show_legend=False)
    _draw_2d_scatter(axes[1], right_frame, x1_key, y1_key, types, right_title, show_legend=False)
    _draw_regression_overlays(
        axes[0],
        left_frame,
        x_key,
        y_key,
        show_ols=show_ols,
        show_ci=show_ci,
        show_lowess=show_lowess,
        confidence=confidence,
    )
    _draw_regression_overlays(
        axes[1],
        right_frame,
        x1_key,
        y1_key,
        show_ols=show_ols,
        show_ci=show_ci,
        show_lowess=show_lowess,
        confidence=confidence,
    )

    if export_regression:
        left_report = fs._regression_report(
            left_frame,
            output_dir,
            x_key,
            y_key,
            plot_name="2d_pairs",
            panel="left",
            confidence=confidence,
        )
        right_report = fs._regression_report(
            right_frame,
            output_dir,
            x1_key,
            y1_key,
            plot_name="2d_pairs",
            panel="right",
            confidence=confidence,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if prebooking_alpha is not None:
        handles.append(Line2D([], [], linestyle="none"))
        labels.append(f"prebooking-alpha = {prebooking_alpha:g}")

    fig.legend(
        handles,
        labels,
        title=", ".join(types),
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        borderaxespad=0.0,
    )

    plt.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))
    plt.savefig(outpng, dpi=600, bbox_inches="tight")
    #   plt.show()
    plt.close(fig)
    return outpng


def plts_cost_tradeoff(
    x_key: str,
    y_key: str,
    frame: pd.DataFrame,
    output_dir: Path,
    date_stamp: str,
    fleet_sizes: Sequence[int] | None = None,
    config: Config | None = None,

) -> Path:
    '''frame: pd.DataFrame
    "lambda",
    "mode_id",
    "fleet_max",
    "avg_net_expenditure",
    "avg_service_time",
    '''
    outpng = Path(output_dir) / f"cost_tradeoff_{date_stamp}.png"
    outpng.parent.mkdir(parents=True, exist_ok=True)

    lambda_values = _ordered_values(frame, "lambda", config.lambdas)
    ht_values = _ordered_values(frame, "ht", config.ht)
    hs_values = _ordered_values(frame, "hs", config.hs)
    fleet_values = _ordered_values(frame, "fleet_max", fleet_sizes)
    mode_values = _ordered_values(frame, "mode_id", config.modes)
    fleet_markers = _fleet_marker_map(fleet_values)
    color_specs = [
    ("lambda", lambda_values, _value_color_map(lambda_values)),
    ("ht", ht_values, _value_color_map(ht_values)),
    ("hs", hs_values, _value_color_map(hs_values)),
]
    fig, axes = plt.subplots(1, 3, figsize=(12, 7),
                           sharex=True, sharey=True)
    fig.supxlabel("Unit expenditure (£/pax)")
    fig.supylabel("Unit time (minutes/pax)")
    fig.suptitle("Cost-time Tradeoff")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    # #折线
    # for mode_id, group in frame.groupby("mode_id"):
    #     sorted_group = group.sort_values(by="lambda")
    #     ax.plot(
    #         sorted_group[x_key],
    #         sorted_group[y_key],
    #         color="black",
    #         linestyle=COST_TRADEOFF_LINESTYLES.get(int(mode_id), "-"),
    #         linewidth=1.8,
    #         alpha=0.85,
    #         label="_nolegend_",
    #         zorder=2,
    #     )
    #散点
    for ax, (color_key, values, colors) in zip(axes, color_specs):
        for fleet_size, group in frame.groupby("fleet_max"):
            point_colors = group[color_key].map(colors).tolist()
            ax.scatter(
                group[x_key],
                group[y_key],
                c=point_colors,
                marker=fleet_markers[int(fleet_size)],
                s=72,
                edgecolors="black",
                linewidths=0.35,
                alpha=0.95,
                label="_nolegend_",
                zorder=3,
            )
        _add_value_legends(
            ax,
            values=values,
            colors=colors,
            title=color_key,
        )

    plt.tight_layout()
    plt.savefig(outpng, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return outpng


def _ordered_values(
    frame: pd.DataFrame,
    column: str,
    preferred_values: Sequence[int] | None,
) -> list:
    present_values = set(frame[column].dropna().tolist())
    if preferred_values is None:
        return sorted(present_values)

    ordered_values = []
    for value in preferred_values:
        candidate = float(value) 
        if candidate in present_values and candidate not in ordered_values:
            ordered_values.append(candidate)
    return ordered_values


def _value_color_map(values: Sequence) -> dict:
    cmap = plt.colormaps.get_cmap("viridis")
    if len(values) == 1:
        return {values[0]: cmap(0.5)}
    return {
        value: cmap(index / (len(values) - 1))
        for index, value in enumerate(values)
    }


def _fleet_marker_map(fleet_values: Sequence[int]) -> dict[int, str]:
    return {
        int(fleet_size): COST_TRADEOFF_MARKERS[index % len(COST_TRADEOFF_MARKERS)]
        for index, fleet_size in enumerate(fleet_values)
    }


def _add_value_legends(ax, values, colors, title) -> None:
    handles = [
        Line2D(
            [],
            [],
            color=colors[value],
            linestyle="-",
            linewidth=2.0,
            label=f"{title} = {_format_number(value)}",
        )
        for value in values
    ]
    ax.legend(handles=handles, title=title)


def _add_mode_legends(
    ax,
    mode_values: Sequence[int] | None = None,
) -> None:
    fig = ax.figure
    legend_x = 0.90
    if mode_values is not None:
        mode_handles = [
            Line2D(
                [],
                [],
                color="black",
                linestyle=COST_TRADEOFF_LINESTYLES.get(int(mode_id), "-"),
                linewidth=2.0,
                label=MODE_LABELS.get(int(mode_id), f"Mode {int(mode_id)}"),
            )
            for mode_id in mode_values
        ]
        fig.legend(
            handles=mode_handles,
            title="mode_id",
            loc="upper left",
            bbox_to_anchor=(legend_x, 0.72),
            bbox_transform=fig.transFigure,
            borderaxespad=0.0,
            framealpha=0.88,
            fontsize=8,
        )

def _add_fleet_legends(
    ax,
    fleet_values: Sequence[int] | None = None,
    fleet_markers: dict[int, str] | None = None,
) -> None:
    fig = ax.figure
    legend_x = 0.90
    if fleet_values is not None:
        fleet_handles = [
            Line2D(
                [],
                [],
                color="black",
                marker=fleet_markers[int(fleet_size)],
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label=f"Fleet Size = {int(fleet_size)}",
        )
        for fleet_size in fleet_values
    ]
        fig.legend(
            handles=fleet_handles,
            title="fleet_max",
            loc="upper left",
            bbox_to_anchor=(legend_x, 0.92),
            bbox_transform=fig.transFigure,
            borderaxespad=0.0,
            framealpha=0.88,
            fontsize=8,
        )

def _format_number(value) -> str:
    numeric_value = float(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:g}"

# 画出未服务请求占比的线条
def _draw_unserved_portion_lines(ax, frame: pd.DataFrame, x_key) -> None: 
    if "acceptance_rate" not in frame.columns:
        raise KeyError("Missing required column for unserved portion lines: acceptance_rate")

    portion_ax = ax.twinx()
    grouped_records: dict[int, list[dict]] = {}

    for row in frame.to_dict(orient="records"):
        mode_id = int(row["mode_id"])
        grouped_records.setdefault(mode_id, []).append(row)

    for mode_id, group_records in grouped_records.items():
        sorted_records = sorted(group_records, key=lambda row: float(row[x_key]))
        x_values = [float(row[x_key]) for row in sorted_records]
        y_values = []
        for row in sorted_records:
            acceptance_rate = float(row["acceptance_rate"])
            portion = 1.0 - acceptance_rate
            y_values.append(portion)

        portion_ax.plot(
            x_values,
            y_values,
            color=MODE_COLORS.get(mode_id, "black"),
            linewidth=1.8,
            alpha=0.9,
            label="_nolegend_",
        )

    portion_ax.set_ylabel("unserved portion of all requests")
    portion_ax.set_ylim(bottom=0.0)


def _draw_2d_scatter(
    ax,
    frame: pd.DataFrame,
    x_key,
    y_key,
    types: list[str],
    title: str,
    show_legend: bool,
) -> None:
    records = frame.to_dict(orient="records")

    grouped_records: dict[tuple, list[dict]] = {}
    for row in records:
        group_key = tuple(row[type_key] for type_key in types)
        grouped_records.setdefault(group_key, []).append(row)

    for group_key, group_records in grouped_records.items(): #row[x_key]
        x_values = [float(row[x_key]) for row in group_records]
        y_values = [float(row[y_key]) for row in group_records]
        scatter_kwargs = {
            "s": 50,
            "alpha": 0.85,
            "label": group_label(types, group_key),
        }

        if types == ["mode_id"]:
            mode_id = int(group_key[0])
            scatter_kwargs["color"] = MODE_COLORS.get(mode_id, "black")
            scatter_kwargs["label"] = MODE_LABELS.get(mode_id, f"Mode {mode_id}")

        ax.scatter(x_values, y_values, **scatter_kwargs)

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if show_legend:
        ax.legend(
            title=", ".join(types),
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            borderaxespad=0.0,
        )


def _draw_regression_overlays(
    ax,
    frame: pd.DataFrame,
    x_key,
    y_key,
    *,
    show_ols: bool,
    show_ci: bool,
    show_lowess: bool,
    confidence: float,
) -> None:
    if show_ci:
        _draw_confidence_bands(ax, frame, x_key, y_key, confidence=confidence)
    if show_ols:
        draw_ols_lines(ax, frame, x_key, y_key, confidence=confidence)
    if show_lowess:
        draw_lowess_lines(ax, frame, x_key, y_key)

MODE_LABELS = {
    1: "Mode 1",
    2: "Mode 2",
    3: "Mode 3",
    4: "Mode 4",
}


def fit_ols_by_mode(    #根据 mode_id 分组进行 OLS 回归拟合
    frame: pd.DataFrame,
    x_key,
    y_key,
    confidence: float = 0.95,
) -> dict[int, dict]:
    required_columns = {"mode_id", x_key, y_key}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise KeyError(f"Missing regression columns: {sorted(missing_columns)}")

    valid_frame = frame.loc[:, ["mode_id", x_key, y_key]].copy()
    valid_frame[x_key] = pd.to_numeric(valid_frame[x_key], errors="coerce")
    valid_frame[y_key] = pd.to_numeric(valid_frame[y_key], errors="coerce")
    valid_frame["mode_id"] = pd.to_numeric(valid_frame["mode_id"], errors="coerce")
    valid_frame = valid_frame.dropna(subset=["mode_id", x_key, y_key])
    valid_frame["mode_id"] = valid_frame["mode_id"].astype(int)

    results: dict[int, dict] = {}
    for mode_id, mode_frame in valid_frame.groupby("mode_id"):
        x_values = mode_frame[x_key].astype(float)
        y_values = mode_frame[y_key].astype(float)
        n = int(len(mode_frame))
        x_min = float(x_values.min()) if n else None
        x_max = float(x_values.max()) if n else None

        result = {
            "status": "insufficient_data",
            "x_grid": np.array([], dtype=float),
            "prediction": pd.DataFrame(),
            "model": None,
            "n": n,
            "x_min": x_min,
            "x_max": x_max,
        }

        if n < 2 or x_values.nunique() < 2:
            results[int(mode_id)] = result
            continue

        x_series = pd.Series(x_values.to_numpy(dtype=float), name=str(x_key))
        y_series = pd.Series(y_values.to_numpy(dtype=float), name=str(y_key))
        design = sm.add_constant(x_series, has_constant="add")
        model = sm.OLS(y_series, design).fit()

        x_grid = np.linspace(x_min, x_max, 100)
        prediction_design = sm.add_constant(
            pd.Series(x_grid, name=str(x_key)),
            has_constant="add",
        )
        prediction = model.get_prediction(prediction_design).summary_frame(
            alpha=1.0 - confidence,
        )

        result.update(
            {
                "status": "ok",
                "x_grid": x_grid,
                "prediction": prediction,
                "model": model,
            }
        )
        results[int(mode_id)] = result

    return results



def draw_ols_lines(
    ax,
    frame: pd.DataFrame,
    x_key,
    y_key,
    confidence: float = 0.95,
) -> None:
    for mode_id, result in fit_ols_by_mode(frame, x_key, y_key, confidence).items():
        if result["status"] != "ok":
            continue
        ax.plot(
            result["x_grid"],
            result["prediction"]["mean"],
            color=MODE_COLORS.get(mode_id, "black"),
            linewidth=2.0,
            linestyle="-",
            alpha=0.95,
            label="_nolegend_",
            zorder=2.5,
        )


def _draw_confidence_bands(
    ax,
    frame: pd.DataFrame,
    x_key,
    y_key,
    confidence: float = 0.95,
) -> None:
    for mode_id, result in fit_ols_by_mode(frame, x_key, y_key, confidence).items():
        if result["status"] != "ok":
            continue
        prediction = result["prediction"]
        ax.fill_between(
            result["x_grid"],
            prediction["mean_ci_lower"].astype(float),
            prediction["mean_ci_upper"].astype(float),
            color=MODE_COLORS.get(mode_id, "black"),
            alpha=0.14,
            linewidth=0,
            label="_nolegend_",
            zorder=1.5,
        )


def draw_lowess_lines(
    ax,
    frame: pd.DataFrame,
    x_key,
    y_key,
    frac: float = 0.6,
) -> None:
    frame[x_key] = pd.to_numeric(frame[x_key], errors="coerce") # 将 x_key 列转换为数值，无法转换的值会变成 NaN
    frame[y_key] = pd.to_numeric(frame[y_key], errors="coerce")
    for mode_id, mode_frame in frame.groupby("mode_id"):
        mode_frame = mode_frame.sort_values(by=x_key)
        if len(mode_frame) < 2 or mode_frame[x_key].nunique() < 2:
            continue

        smoothed = lowess(
            mode_frame[y_key].astype(float),
            mode_frame[x_key].astype(float),
            frac=frac,
            return_sorted=True,
        )
        ax.plot(
            smoothed[:, 0],
            smoothed[:, 1],
            color=MODE_COLORS.get(int(mode_id), "black"),
            linewidth=1.8,
            linestyle="--",
            alpha=0.9,
            label="_nolegend_",
            zorder=2.25,
        )

def _regression_timestamp(timestamp: str | None) -> str:
    global _DEFAULT_REGRESSION_TIMESTAMP
    if timestamp is not None:
        return timestamp
    if _DEFAULT_REGRESSION_TIMESTAMP is None:
        _DEFAULT_REGRESSION_TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M%S")
    return _DEFAULT_REGRESSION_TIMESTAMP


# 2x2 subplots
def plts_4s(path_file: Path,
            x_key, y_key, z_key,
            offset: bool) -> Path:
    # ===== 1. 读取 JSON 文件 =====
    records, json_path = load_records(Path(path_file))
    outpng = Path(json_path).with_name("4s_" + json_path.stem).with_suffix(".png")



    # ===== 3. 创建 2x2 图 =====
    fig = plt.figure(figsize=(14, 10))
    ax_3d = fig.add_subplot(221, projection="3d")
    ax_xy = fig.add_subplot(222)
    ax_xz = fig.add_subplot(223)
    ax_yz = fig.add_subplot(224)

    ax_3d = model_3d(ax_3d, records, x_key, y_key, z_key, offset)
    ax_xy = add_projections(ax_xy, records, 
                 x_key, y_key, z_key,
                 'xy',
                 )
    ax_xz = add_projections(ax_xz, records, 
                 x_key, y_key, z_key,
                 'xz',
                 )
    ax_yz = add_projections(ax_yz, records, 
                 x_key, y_key, z_key,
                 'yz',
                 )    

    # ===== 4. 坐标轴和标题 =====
    x_min, x_max = -0.08, 1.05
    y_min, y_max = -0.08, 1.05
    z_min, z_max = 18, 62

    #2x2定位
    ax_3d.view_init(elev=24, azim=-58)

    # XY
    ax_xy.set_xlim(x_min, x_max)
    ax_xy.set_ylim(y_min, y_max)
    ax_xy.set_xticks([0.0, 0.5, 1.0])
    ax_xy.set_yticks([0.0, 0.5, 1.0])
    ax_xy.set_xlabel("temporal")
    ax_xy.set_ylabel("spatial")
    ax_xy.set_title("temporal-spatial Projection")
    ax_xy.grid(True, alpha=0.3)

    # XZ
    ax_xz.set_xlim(x_min, x_max)
    ax_xz.set_ylim(z_min, z_max)
    ax_xz.set_xticks([0.0, 0.5, 1.0])
    ax_xz.set_yticks([20, 40, 60])
    ax_xz.set_xlabel("temporal")
    ax_xz.set_ylabel("lambda")
    ax_xz.set_title("temporal-lambda Projection")
    ax_xz.grid(True, alpha=0.3)

    # YZ
    ax_yz.set_xlim(y_min, y_max)
    ax_yz.set_ylim(z_min, z_max)
    ax_yz.set_xticks([0.0, 0.5, 1.0])
    ax_yz.set_yticks([20, 40, 60])
    ax_yz.set_xlabel("spatial")
    ax_yz.set_ylabel("lambda")
    ax_yz.set_title("spatial-lambda Projection")
    ax_yz.grid(True, alpha=0.3)

    # ===== 5. 图例 =====
    handles, labels = ax_3d.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="mode_id",
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        ncol=1,
        borderaxespad=0.0,
    )

    plt.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))

    plt.savefig(outpng, dpi=600, bbox_inches="tight")
    # plt.show()
    # plt.pause(0.1)  # 确保图像显示出来
    #plt.close(fig)
    return outpng

# 3d scatter plot
def model_3d(ax, records, x_key, y_key, z_key,
             offset_):
    # 为避免同一场景的点完全重叠，给不同 mode 一个很小的偏移
    if offset_:
        offsets = {
            1: (-0.03, -0.03, -0.8),
            2: (-0.03,  0.03, -0.3),
            3: ( 0.03, -0.03,  0.3),
            4: ( 0.03,  0.03,  0.8),
        }
    else:
        offsets = {}
    add_scatter(ax, records, 
                x_key, y_key, z_key,
                offsets,
                )

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_zlabel(z_key)
    # ax.set_title("Optimal Modes Across Selected Scenarios")
    return ax

def add_surface(ax, x_values: list[float], y_values: list[float], z_values: list[float]) -> bool:
    if len(x_values) < 3:
        print("Not enough optimal points to build a 3D surface. Falling back to scatter only.")
        return False

    try:
        surface = ax.plot_trisurf(
            x_values,
            y_values,
            z_values,
            cmap="viridis",
            linewidth=0.4,
            antialiased=False,
            shade=True,
        )
    except Exception as exc:
        print(f"Unable to generate 3D surface ({exc}). Falling back to scatter only.")
        return False

    colorbar = plt.colorbar(surface, ax=ax, shrink=0.68, pad=0.1)
    colorbar.set_label("total_requests")
    return True


def add_scatter(ax, records: pd.DataFrame,
                x_key, y_key, z_key = None,
                offsets: dict[int, tuple[float, float, float]] | None = None,
                ) -> None:
    if offsets is None:
        offsets = {}
    added_labels: set[int] = set()

    if z_key is None: # 2D散点图
        x = x_key
        y = y_key
        rng = np.random.default_rng(0)
        dx = np.round(
            rng.uniform(-0.05, 0.05, size=x.shape),
            3,
        )
        dy = np.round(
            rng.uniform(-0.05, 0.05, size=y.shape),
            3,
        )
        ax.scatter(
            x + dx,
            y + dy,
            s=22,
            marker="o",
            color="tab:blue",
            label="requests",
            edgecolors="black",
            linewidths=0.4,
            alpha=0.5,
            zorder=5,
        )
    else:# 3D散点图
        for _, row in records.iterrows():
            mode_id = int(row["mode_id"])
            x = float(row[x_key])
            y = float(row[y_key])
            z = float(row[z_key])
            dx, dy, dz = offsets.get(mode_id, (0.0, 0.0, 0.0))
            label = MODE_LABELS[mode_id] if mode_id not in added_labels else None
            if label is not None:
                    added_labels.add(mode_id)

            ax.scatter(
                x + dx,
                y + dy,
                z + dz,
                color=MODE_COLORS.get(mode_id, "black"),
                edgecolors="black",
                linewidths=0.4,
                s=60,
                depthshade=False,
                label=label,
                )
    return ax

def add_projections(ax, records: pd.DataFrame, 
                    x_key = None, y_key = None, z_key = None, 
                    plane = None,
                    offsets: dict[int, tuple[float, float, float]] | None = None,
                    ) -> None:
    if offsets is None:
        offsets = {}
    for _, row in records.iterrows():
        mode_id = row["mode_id"]
        x = row[x_key]
        y = row[y_key]
        z = row[z_key]
        dx, dy, dz = offsets.get(mode_id, (0.0, 0.0, 0.0))
        x0 = x + dx
        y0 = y + dy
        z0 = z + dz
        color = MODE_COLORS.get(mode_id, "black")
        label = MODE_LABELS.get(mode_id, f"Mode {mode_id}")

        if plane == "xy": # XY projection
            a = x0,
            b = y0,
        elif plane == "xz": # XZ projection
            a = x0,
            b = z0,
        elif plane == "yz": # YZ projection
            a = y0,
            b = z0,
        
        ax.scatter(
            a,
            b,
            color=MODE_COLORS.get(mode_id, "black"),
            s=50,
            alpha=0.85,
        )
    return ax

def group_label(types: list[str], group_key: tuple) -> str:
    return ", ".join(
        f"{type_key}={type_value}"
        for type_key, type_value in zip(types, group_key)
    )

def load_records(path_file: Path) -> tuple[list[dict], Path]:
    results_path = Path(path_file)
    if not results_path.exists():
        raise FileNotFoundError(
            "file not found. Run the scenario export first, "
            "then rerun test.py."
        )

    with results_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not records:
        raise ValueError("optimal_modes() returned an empty result set.")

    return records, results_path

def demand_3d(frame: pd.DataFrame,  #x_key, y_key, z_key 分别是 config.ht, config.hs, config.lambdas,但是legend按照 mode_id 来显示颜色。
            output_dir: Path,
            date_stamp: str,
            x_key: str, y_key: str, z_key: str,
            offset: bool) -> Path:
    outpng = Path(output_dir) / f"demand_3d_{date_stamp}.png"

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    model_3d(ax, frame, x_key, y_key, z_key, offset)
    ax.legend(
        title="mode_id",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    plt.tight_layout(rect=(0.0, 0.0, 0.72, 1.0))
    plt.savefig(outpng, dpi=600, bbox_inches="tight")
    plt.show()
    #plt.close(fig)
    return outpng


def draw_section(
    frame: pd.DataFrame,
    x_key: str,
    y_key: str,
    z_key: str,
    offset: bool = False,
    *,
    section_axes: Sequence[str] | None = None,
    section_values: dict[str, Sequence[float]] | None = None,
    max_sections: int = 9,
    max_cols: int = 3,
    section_strategy: str = "quantile",
    tolerance: float = 1e-9,
) -> list[Path]:
    required_columns = {x_key, y_key, z_key, "mode_id"}
    missing_columns = required_columns - set(frame.columns)
    if isinstance(offset, (tuple, list)):
        offset = bool(offset[0]) if len(offset) == 1 else bool(offset)

    axis_keys = [x_key, y_key, z_key]
    axis_lookup = {
        "x": x_key,
        "y": y_key,
        "z": z_key,
        x_key: x_key,
        y_key: y_key,
        z_key: z_key,
    }

    if section_axes is None:
        selected_axes = axis_keys
    else:
        selected_axes = []
        for axis in section_axes:
            if axis not in axis_lookup:
                raise ValueError(
                    f"Unknown section axis '{axis}'. Use one of {sorted(axis_lookup)}."
                )
            selected_axes.append(axis_lookup[axis])

    offsets = {
        1: (-0.03, -0.03, -0.8),
        2: (-0.03, 0.03, -0.3),
        3: (0.03, -0.03, 0.3),
        4: (0.03, 0.03, 0.8),
    } if offset else {}

    offset_by_axis = {
        x_key: {mode_id: values[0] for mode_id, values in offsets.items()},
        y_key: {mode_id: values[1] for mode_id, values in offsets.items()},
        z_key: {mode_id: values[2] for mode_id, values in offsets.items()},
    }

    def _select_section_values(section_key: str) -> list[float]:
        if section_values is not None and section_key in section_values:
            return [float(value) for value in section_values[section_key]]

        numeric_values = pd.to_numeric(frame[section_key], errors="coerce").dropna()
        unique_values = np.asarray(sorted(numeric_values.unique()), dtype=float)
        if unique_values.size == 0:
            return []
        if unique_values.size <= max_sections:
            return unique_values.tolist()

        if section_strategy == "quantile":
            targets = np.quantile(unique_values, np.linspace(0.0, 1.0, max_sections))
            indices = [
                int(np.abs(unique_values - target).argmin())
                for target in targets
            ]
        elif section_strategy == "even":
            indices = np.rint(
                np.linspace(0, unique_values.size - 1, max_sections)
            ).astype(int).tolist()
        else:
            raise ValueError(
                "section_strategy must be 'quantile' or 'even'. "
                "Use section_values for manual selection."
            )

        return unique_values[sorted(set(indices))].tolist()

    def _slice_frame(section_key: str, section_value: float) -> pd.DataFrame:
        numeric_values = pd.to_numeric(frame[section_key], errors="coerce")
        mask = np.isclose(
            numeric_values.to_numpy(dtype=float),
            float(section_value),
            atol=tolerance,
            rtol=0.0,
            equal_nan=False,
        )
        return frame.loc[mask].copy()

    def _format_section_value(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return f"{value:g}"

    def _draw_section_scatter(
        ax,
        section_frame: pd.DataFrame,
        plot_x: str,
        plot_y: str,
    ) -> None:
        if section_frame.empty:
            ax.text(
                0.5,
                0.5,
                "no data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        for mode_id, mode_frame in section_frame.groupby("mode_id"):
            mode_id = int(mode_id)
            x_values = pd.to_numeric(mode_frame[plot_x], errors="coerce").astype(float)
            y_values = pd.to_numeric(mode_frame[plot_y], errors="coerce").astype(float)
            if offset:
                x_values = x_values + offset_by_axis[plot_x].get(mode_id, 0.0)
                y_values = y_values + offset_by_axis[plot_y].get(mode_id, 0.0)
            ax.scatter(
                x_values,
                y_values,
                color=MODE_COLORS.get(mode_id, "black"),
                edgecolors="black",
                linewidths=0.4,
                s=50,
                alpha=0.85,
            )

    outpngs: list[Path] = []
    for section_key in selected_axes:
        plot_x, plot_y = [key for key in axis_keys if key != section_key]
        values = _select_section_values(section_key)
        if not values:
            continue

        n_sections = len(values)
        n_cols = min(max_cols, n_sections)
        n_rows = int(np.ceil(n_sections / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.0 * n_cols, 4.0 * n_rows),
            squeeze=False,
        )
        flat_axes = axes.ravel()

        for ax, section_value in zip(flat_axes, values):
            section_frame = _slice_frame(section_key, section_value)
            _draw_section_scatter(ax, section_frame, plot_x, plot_y)
            ax.set_xlabel(plot_x)
            ax.set_ylabel(plot_y)
            ax.set_title(f"{section_key} = {_format_section_value(section_value)}")
            ax.grid(True, alpha=0.3)

        for ax in flat_axes[n_sections:]:
            ax.set_visible(False)

        modes = sorted(
            pd.to_numeric(frame["mode_id"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="white",
                markerfacecolor=MODE_COLORS.get(mode_id, "black"),
                markeredgecolor="black",
                markersize=8,
                linestyle="",
                label=MODE_LABELS.get(mode_id, f"Mode {mode_id}"),
            )
            for mode_id in modes
        ]
        fig.legend(
            handles=handles,
            title="mode_id",
            loc="center left",
            bbox_to_anchor=(0.90, 0.5),
            borderaxespad=0.0,
        )
        fig.suptitle(f"Section by {section_key}: {plot_x}-{plot_y}", y=0.995)
        plt.tight_layout(rect=(0.0, 0.0, 0.86, 0.96))

        plt.show()

    return 

#导入class
from main import config

if __name__ == "__main__":    
    results_dir = Path(__file__).resolve().parent /"rs" 
    result_file = results_dir / "de_optimal_260630_1337.json" # 路径
    parts = result_file.stem.split("_")
    date_stamp = "_".join(parts[-2:])
    results_frame = pd.read_json(result_file) 
    types = ["mode_id"] 
    # plot_columns = [
    #     "mode_id",
    #     "served_requests",
    #     "net_expenditure",
    #     "total_service_time",
    # ]
    
    a = results_frame[["net_expenditure", "total_service_time"]].copy() # dividend
    #计算请求的平均值
    fr_plot1 = dg.avg_served(results_frame, a, "acceptance") 
    
    # fs.json_to_excel(output_dir/ "1_rs_260519_2037.json")
    output_dir = Path(__file__).resolve().parent / "rs"/"rs_plot" #路径
    plts_2d(fr_plot1,output_dir,date_stamp,"avg_net_expenditure","avg_service_time",types)
    plts_2d_pair(   #画图 #plts_3d xyz图, plts_2d xy图, plts_4s 2x2图
        fr_plot1,
        fr_plot1,
        output_dir,
        date_stamp,
        "served_requests",
        "served_requests",
        "avg_net_expenditure",
        "avg_service_time",
        types,
        left_title="avg_net_expenditure",
        right_title="avg_service_time",
        prebooking_alpha=config.pre_alpha,
    ) 
    

    fr_plot2 = dg.avg_served(results_frame, a, "acceptance") #计算请求的平均值
    plts_cost_tradeoff("avg_net_expenditure","avg_service_time", fr_plot2, output_dir, date_stamp, config = config )
    if "optimal" in str(result_file): # 如果是最优解的结果文件，则画 3D 图和需求图
        offset = True
        demand_3d(fr_plot2, output_dir, date_stamp, "ht", "hs", "lambda", offset)
        draw_section(
        fr_plot2,
        "ht",
        "hs",
        "lambda",
        offset,
        )
    else:
        offset = True
    
    plts_3d(fr_plot2, output_dir, date_stamp, "avg_net_expenditure", "avg_service_time", "lambda", offset)  
    


    # print("Processing complete.")
    # input("Press Enter to continue...")
    # print("Continuing program...")

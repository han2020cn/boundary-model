from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess


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


def plt_surface(
    frame: pd.DataFrame,
    output_path: Path,
    axes: tuple[str, str],
    z: str,
) -> Path:
    
    if len(axes) != 2:
        raise ValueError("axes must contain exactly two column names: (x, y)")

    x, y = axes
    requested_columns = [x, y, z]
    missing_columns = [key for key in requested_columns if key not in frame.columns]
    if missing_columns:
        raise KeyError(f"frame is missing columns: {missing_columns}")
    if len(set(requested_columns)) != 3:
        raise ValueError("x, y, and z must refer to three different columns")

    surface_data = frame[requested_columns].apply(pd.to_numeric, errors="coerce")
    surface_data = surface_data.dropna().groupby([x, y], as_index=False)[z].max()
    if len(surface_data) < 3:
        raise ValueError("at least three distinct, valid x-y points are required")

    x_values = surface_data[x].to_numpy(dtype=float, copy=True)
    y_values = surface_data[y].to_numpy(dtype=float, copy=True)
    z_values = surface_data[z].to_numpy(dtype=float, copy=True)

    triangulation = mtri.Triangulation(x_values, y_values)
    if triangulation.triangles.size == 0:
        raise ValueError("x-y points must not all be collinear")

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_values.min(), x_values.max(), 200),
        np.linspace(y_values.min(), y_values.max(), 200),
    )
    interpolator = mtri.CubicTriInterpolator(triangulation, z_values, kind="geom")
    grid_z = interpolator(grid_x, grid_y).filled(np.nan)

    output_path = Path(output_path)
    outpng = output_path

    outpng.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        grid_x,
        grid_y,
        grid_z,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rcount=200,
        ccount=200,
        alpha=0.92,
    )
    ax.scatter(
        x_values,
        y_values,
        z_values,
        color="black",
        s=10,
        alpha=0.55,
        depthshade=False,
        label="observations",
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(f"{z} surface over {x} and {y}")
    ax.view_init(elev=28, azim=-135)
    ax.legend(loc="upper right")
    fig.colorbar(surface, ax=ax, shrink=0.68, pad=0.1, label=z)
    fig.tight_layout()
    fig.savefig(outpng, dpi=600, bbox_inches="tight")
    plt.show()
    return outpng

def plt_scatter_2d(
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
    """2d scatter plot: xy + regression"""
    outpng = Path(output_path)/f"scatter_2d_{date_stamp}.png"
    outpng.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    _draw_2d_scatter(ax, frame, x_key, y_key, types, "x-y scatter plot", show_legend=True)
    # _draw_regression_overlays(
    #     ax,
    #     frame,
    #     x_key,
    #     y_key,
    #     show_ols=show_ols,
    #     show_ci=show_ci,
    #     show_lowess=show_lowess,
    #     confidence=confidence,
    # )
    # # 导出回归结果
    # if export_regression: 
    #     report_frame = fs._regression_report(
    #         frame,
    #         output_path,
    #         x_key,
    #         y_key,
    #         plot_name="2d",
    #         panel="main",
    #         confidence=confidence,
    #     )


    plt.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    plt.savefig(outpng, dpi=600, bbox_inches="tight")
    # plt.show()
    # plt.close(fig)
    return outpng

# a pair of 2d scatter plots: 相同的x轴，不同的y轴，regression
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
    prebooking_alpha: float | None = None,
    show_ols: bool = True,
    show_ci: bool = True,
    show_lowess: bool = False,
    export_regression: bool = True,
    confidence: float = 0.95,
    regression_timestamp: str | None = None,
) -> Path:
    outpng = Path(output_dir) / f"2d_pair_{date_stamp}.png"
    outpng.parent.mkdir(parents=True, exist_ok=True)
    left_title = y_key
    right_title = y1_key
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False, sharey=False)
    _draw_2d_scatter(axes[0], left_frame, x_key, y_key, types, left_title, show_legend=False)
    _draw_2d_scatter(axes[1], right_frame, x1_key, y1_key, types, right_title, show_legend=False)
    # _draw_regression_overlays(
    #     axes[0],
    #     left_frame,
    #     x_key,
    #     y_key,
    #     show_ols=show_ols,
    #     show_ci=show_ci,
    #     show_lowess=show_lowess,
    #     confidence=confidence,
    # )
    # _draw_regression_overlays(
    #     axes[1],
    #     right_frame,
    #     x1_key,
    #     y1_key,
    #     show_ols=show_ols,
    #     show_ci=show_ci,
    #     show_lowess=show_lowess,
    #     confidence=confidence,
    # )

    # if export_regression:
    #     left_report = fs._regression_report(
    #         left_frame,
    #         output_dir,
    #         x_key,
    #         y_key,
    #         plot_name="2d_pairs",
    #         panel="left",
    #         confidence=confidence,
    #     )
    #     right_report = fs._regression_report(
    #         right_frame,
    #         output_dir,
    #         x1_key,
    #         y1_key,
    #         plot_name="2d_pairs",
    #         panel="right",
    #         confidence=confidence,
    #     )

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


def plts_tradeoff(
    x_key: str,
    y_key: str,
    frame: pd.DataFrame,
    output_dir: Path,
    date_stamp: str,
    colors: str,
    markers: str | None = None,
    linestyles: str | None = None,
) -> Path:

    outpng = Path(output_dir) / f"tradeoff_{date_stamp}.png"
    outpng.parent.mkdir(parents=True, exist_ok=True)

    requested_columns = [x_key, y_key, colors]
    requested_columns.extend(key for key in (markers, linestyles) if key is not None)
    missing_columns = [key for key in requested_columns if key not in frame.columns]
    if missing_columns:
        raise KeyError(f"frame is missing columns: {missing_columns}")

    color_values = _ordered_values(frame, colors, None)
    marker_values = _ordered_values(frame, markers, None) if markers is not None else []
    linestyle_values = (
        _ordered_values(frame, linestyles, None) if linestyles is not None else []
    )
    color_map = _value_color_map(color_values)
    marker_map = _value_marker_map(marker_values)
    linestyle_map = _value_linestyle_map(linestyle_values)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title("Cost-time Tradeoff")
    ax.grid(True, alpha=0.3)

    if linestyles is not None:
        for style_value, group in frame.groupby(linestyles, sort=False):
            sorted_group = group.sort_values(by=x_key)
            ax.plot(
                sorted_group[x_key],
                sorted_group[y_key],
                color="black",
                linestyle=linestyle_map[style_value],
                linewidth=1.8,
                alpha=0.85,
                label="_nolegend_",
                zorder=2,
            )

    if markers is None:
        scatter_groups = [(None, frame)]
    else:
        scatter_groups = frame.groupby(markers, sort=False)

    for marker_value, group in scatter_groups:
        ax.scatter(
            group[x_key],
            group[y_key],
            c=group[colors].map(color_map).tolist(),
            marker="o" if markers is None else marker_map[marker_value],
            s=72,
            edgecolors="black",
            linewidths=0.35,
            alpha=0.95,
            label="_nolegend_",
            zorder=3,
        )

    _add_cost_tradeoff_legends(
        ax,
        color_key=colors,
        color_values=color_values,
        color_map=color_map,
        marker_key=markers,
        marker_values=marker_values,
        marker_map=marker_map,
        linestyle_key=linestyles,
        linestyle_values=linestyle_values,
        linestyle_map=linestyle_map,
    )

    plt.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
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




def _value_marker_map(values: Sequence) -> dict:
    return {
        value: COST_TRADEOFF_MARKERS[index % len(COST_TRADEOFF_MARKERS)]
        for index, value in enumerate(values)
    }


def _value_linestyle_map(values: Sequence) -> dict:
    styles = tuple(COST_TRADEOFF_LINESTYLES.values())
    return {value: styles[index % len(styles)] for index, value in enumerate(values)}


def _add_cost_tradeoff_legends(
    ax,
    *,
    color_key: str,
    color_values: Sequence,
    color_map: dict,
    marker_key: str | None,
    marker_values: Sequence,
    marker_map: dict,
    linestyle_key: str | None,
    linestyle_values: Sequence,
    linestyle_map: dict,
) -> None:
    color_handles = [
        Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor=color_map[value],
            markeredgecolor="black",
            markersize=7,
            label=f"{color_key} = {_format_number(value)}",
        )
        for value in color_values
    ]
    color_legend = ax.legend(
        handles=color_handles,
        title=color_key,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    ax.add_artist(color_legend)

    style_handles = []
    if marker_key is not None:
        style_handles.extend(
            Line2D(
                [],
                [],
                color="black",
                linestyle="none",
                marker=marker_map[value],
                markerfacecolor="white",
                markersize=7,
                label=f"{marker_key} = {_format_number(value)}",
            )
            for value in marker_values
        )
    if linestyle_key is not None:
        style_handles.extend(
            Line2D(
                [],
                [],
                color="black",
                linestyle=linestyle_map[value],
                linewidth=1.8,
                label=f"{linestyle_key} = {_format_number(value)}",
            )
            for value in linestyle_values
        )
    if style_handles:
        ax.legend(
            handles=style_handles,
            title=" / ".join(
                key for key in (marker_key, linestyle_key) if key is not None
            ),
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            borderaxespad=0.0,
        )


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

def plt_scatter_3d(frame: pd.DataFrame,  #x_key, y_key, z_key 分别是 config.ht, config.hs, config.lambdas,但是legend按照 mode_id 来显示颜色。
            output_dir: Path,
            date_stamp: str,
            x_key: str, y_key: str, z_key: str,
            offset: bool) -> Path:
    outpng = Path(output_dir) / f"scatter_3d_{date_stamp}.png"

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


def plts_section(
    frame: pd.DataFrame,
    x_key: str,
    y_key: str,
    z_key: str,
    offset: bool,
    output_dir: Path,
    *,
    max_sections: int = 12,
    section_axes: Sequence[str] | None = None,
    section_values: dict[str, Sequence[float]] | None = None,
    max_cols: int = 3,
    section_strategy: str = "quantile",
) -> list[Path]:
    tolerance = float(1e-9),
    required_columns = {x_key, y_key, z_key, "mode_id"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise KeyError(f"frame is missing columns: {sorted(missing_columns)}")

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
        selected_axes = [z_key]
    else:
        selected_axes = []
        for axis in section_axes:
            if axis not in axis_lookup:
                raise ValueError(
                    f"Unknown section axis '{axis}'. Use one of {sorted(axis_lookup)}."
                )
            selected_axes.append(axis_lookup[axis])

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

    def _ordered_values(key: str) -> list[float]:
        values = pd.to_numeric(frame[key], errors="coerce").dropna().unique()
        return np.asarray(sorted(values), dtype=float).tolist()

    def _cell_mode(
        section_frame: pd.DataFrame,
        plot_x: str,
        plot_y: str,
        x_value: float,
        y_value: float,
    ) -> int | None:
        numeric_x = pd.to_numeric(section_frame[plot_x], errors="coerce")
        numeric_y = pd.to_numeric(section_frame[plot_y], errors="coerce")
        mask = np.isclose(
            numeric_x.to_numpy(dtype=float), x_value,
            atol=tolerance, rtol=0.0, equal_nan=False,
        ) & np.isclose(
            numeric_y.to_numpy(dtype=float), y_value,
            atol=tolerance, rtol=0.0, equal_nan=False,
        )
        modes = (
            pd.to_numeric(section_frame.loc[mask, "mode_id"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        if len(modes) > 1:
            raise ValueError(
                "draw_section requires one mode per scenario cell; "
                f"found modes {sorted(modes.tolist())} at "
                f"{plot_x}={x_value:g}, {plot_y}={y_value:g}."
            )
        return int(modes[0]) if len(modes) == 1 else None

    outpngs: list[Path] = []
    # Kept in the signature for compatibility with the former scatter layout.
    _ = offset

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
            marker="s",
            color="white",
            markerfacecolor=MODE_COLORS.get(mode_id, "black"),
            markeredgecolor="black",
            markersize=10,
            linestyle="",
            label=MODE_LABELS.get(mode_id, f"Mode {mode_id}"),
        )
        for mode_id in modes
    ]

    for section_key in selected_axes:
        plot_x, plot_y = [key for key in axis_keys if key != section_key]
        section_values_for_axis = _select_section_values(section_key)
        if not section_values_for_axis:
            continue

        x_values = _ordered_values(plot_x)
        y_values = _ordered_values(plot_y)
        if not x_values or not y_values:
            continue

        n_sections = len(section_values_for_axis)
        n_cols = min(max_cols, n_sections)
        n_rows = int(np.ceil(n_sections / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.4 * n_cols + 2.2, 3.0 * n_rows + 1.1),
            squeeze=False,
        )
        flat_axes = axes.ravel()

        for panel_index, (ax, section_value) in enumerate(
            zip(flat_axes, section_values_for_axis)
        ):
            section_frame = _slice_frame(section_key, section_value)
            for row_index, y_value in enumerate(y_values):
                for col_index, x_value in enumerate(x_values):
                    mode_id = _cell_mode(
                        section_frame, plot_x, plot_y, x_value, y_value
                    )
                    if mode_id is not None:
                        ax.add_patch(
                            plt.Rectangle(
                                (col_index - 0.5, row_index - 0.5),
                                1.0,
                                1.0,
                                facecolor=MODE_COLORS.get(mode_id, "black"),
                                edgecolor="none",
                            )
                        )
                        # ax.text(
                        #     col_index,
                        #     row_index,
                        #     f"M{mode_id}",
                        #     color="white" if mode_id in {2, 4} else "black",
                        #     fontsize=7,
                        #     fontweight="bold",
                        #     ha="center",
                        #     va="center",
                        # )

            ax.set_xlim(-0.5, len(x_values) - 0.5)
            ax.set_ylim(len(y_values) - 0.5, -0.5)
            ax.set_xticks(range(len(x_values)))
            ax.set_yticks(range(len(y_values)))
            ax.set_xticklabels(
                [_format_section_value(value) for value in x_values], fontsize=7
            )
            ax.set_yticklabels(
                [_format_section_value(value) for value in y_values], fontsize=7
            )
            ax.set_xticks(np.arange(-0.5, len(x_values), 1.0), minor=True)
            ax.set_yticks(np.arange(-0.5, len(y_values), 1.0), minor=True)
            ax.grid(which="minor", color="0.75", linewidth=0.6)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.set_xlabel(plot_x, fontsize=8)
            ax.set_ylabel(plot_y, fontsize=8)
            ax.set_title(
                f"{section_key} = {_format_section_value(section_value)}",
                fontsize=10,
                pad=6,
            )
            ax.set_aspect("equal")
            for spine in ax.spines.values():
                spine.set_color("0.25")
                spine.set_linewidth(0.8)

        for ax in flat_axes[n_sections:]:
            ax.set_visible(False)

        fig.suptitle(
            f"Section by {section_key}: {plot_x}-{plot_y}",
            fontsize=14,
            y=1.2,
        )
        fig.legend(
            handles=handles,
            title="mode_id",
            loc="center left",
            bbox_to_anchor=(0.86, 0.5),
            borderaxespad=0.0,
            frameon=True,
            edgecolor="black",
        )
        fig.subplots_adjust(
            left=0.07,
            right=0.83,
            bottom=0.07,
            top=0.92,
            wspace=0.30,
            hspace=0.35,
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        outpng = output_path / f"section_by_{section_key}.png"
        fig.savefig(outpng, dpi=600, bbox_inches="tight")
        outpngs.append(outpng)

        plt.show()
        plt.close(fig)

    return outpngs

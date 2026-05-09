from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
# from mpl_toolkits.mplot3d import Axes3D

# ===== 2. 为不同 mode_id 定义颜色 =====
MODE_COLORS = {
    1: "tab:grey",
    2: "tab:blue",
    3: "tab:orange",
    4: "tab:red",
}

MODE_LABELS = {
    1: "Mode 1",
    2: "Mode 2",
    3: "Mode 3",
    4: "Mode 4",
}

COST_TRADEOFF_LINESTYLES = {
    1: "-",
    2: "--",
    3: (0, (6, 2, 1, 2)),
    4: ":",
}

COST_TRADEOFF_MARKERS = ("*", "X", "^", "D", "o", "s", "P")

REGRESSION_REPORT_COLUMNS = [
    "plot_name",
    "panel",
    "x_key",
    "y_key",
    "mode_id",
    "mode_label",
    "n",
    "slope",
    "intercept",
    "r_squared",
    "p_value",
    "confidence_level",
    "x_min",
    "x_max",
    "status",
]

_DEFAULT_REGRESSION_TIMESTAMP: str | None = None

#3d scatter plot
def plts_3d(path_file: Path,
            x_key, y_key, z_key) -> Path:
    records, json_path = load_records(Path(path_file))
    outpng = json_path.with_name("3d_" + json_path.stem).with_suffix(".png")

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    model_3d(ax, records, x_key, y_key, z_key)
    ax.legend(
        title="mode_id",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    plt.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    plt.savefig(outpng, dpi=600, bbox_inches="tight")
    plt.show()
    #plt.close(fig)
    return outpng

# 2d scatter plot
def plts_2d(
    frame: pd.DataFrame,
    output_path: Path,
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
    outpng = Path(output_path)/"2d_plots.png"
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

    if export_regression:
        report_frame = build_regression_report(
            frame,
            x_key,
            y_key,
            plot_name="2d",
            panel="main",
            confidence=confidence,
        )
        export_regression_report(
            [report_frame],
            outpng.parent,
            timestamp=regression_timestamp,
        )

    plt.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    plt.savefig(outpng, dpi=600, bbox_inches="tight")
    plt.show()
    #plt.close(fig)
    return outpng


def plts_2d_pair(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    output_dir: Path,
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
    outpng = Path(output_dir) / f"2d_pairs_{datetime.now().strftime('%m%d_%H%M%S')}.png"
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
    _draw_unserved_portion_lines(axes[1], right_frame, x1_key)

    if export_regression:
        left_report = build_regression_report(
            left_frame,
            x_key,
            y_key,
            plot_name="2d_pairs",
            panel="left",
            confidence=confidence,
        )
        right_report = build_regression_report(
            right_frame,
            x1_key,
            y1_key,
            plot_name="2d_pairs",
            panel="right",
            confidence=confidence,
        )
        export_regression_report(
            [left_report, right_report],
            outpng.parent,
            timestamp=regression_timestamp,
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
    plt.show()
    #plt.close(fig)
    return outpng


def plts_cost_tradeoff(
    frame: pd.DataFrame, 
    output_dir: Path,
    fleet_sizes: Sequence[int] | None = None,
    config: Config | None = None,
    x_key: str = "avg_net_expenditure",
    y_key: str = "avg_service_time",
    color_key: str = "acceptance_rate",
    acceptance_vmin: float = 0.8,
    acceptance_vmax: float = 1.0,
    show: bool = True,
) -> Path:
    '''frame: pd.DataFrame
    "lambda",
    "mode_id",
    "fleet_size",
    "avg_net_expenditure",
    "avg_service_time",
    "acceptance_rate",
    '''
    outpng = Path(output_dir) / f"cost_tradeoff_{datetime.now().strftime('%m%d_%H%M')}.png"
    outpng.parent.mkdir(parents=True, exist_ok=True)

    plot_frame = _valid_cost_tradeoff_frame(frame, x_key, y_key, color_key)
    if fleet_sizes is not None:
        plot_frame = plot_frame[plot_frame["fleet_size"].isin([int(value) for value in fleet_sizes])]
    if config.lambdas is not None:
        plot_frame = plot_frame[plot_frame["lambda"].isin([float(value) for value in config.lambdas])]
    if config.modes is not None:
        plot_frame = plot_frame[plot_frame["mode_id"].isin([int(value) for value in config.modes])]

    if plot_frame.empty:
        raise ValueError("No cost tradeoff rows available after filtering.")

    lambda_values = _ordered_values(plot_frame, "lambda", config.lambdas)
    fleet_values = _ordered_values(plot_frame, "fleet_size", fleet_sizes)
    mode_values = _ordered_values(plot_frame, "mode_id", config.modes)
    lambda_colors = _value_color_map(lambda_values)
    fleet_markers = _fleet_marker_map(fleet_values)

    fig, ax = plt.subplots(figsize=(12, 7))
    acceptance_cmap = plt.get_cmap("RdYlGn")
    norm = Normalize(vmin=acceptance_vmin, vmax=acceptance_vmax)
    #折线
    # for mode_id, group in plot_frame.groupby("mode_id"):
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

    for fleet_size, group in plot_frame.groupby("fleet_size"):
        point_colors = group["lambda"].map(lambda value: lambda_colors[float(value)]).tolist() #根据 lambda 值映射颜色
        ax.scatter(
            group[x_key],
            group[y_key],
            c=point_colors,
            # cmap=acceptance_cmap,
            norm=norm,
            marker=fleet_markers[int(fleet_size)],
            s=72,
            edgecolors="black",
            linewidths=0.35,
            alpha=0.95,
            label="_nolegend_",
            zorder=3,
        )

    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=acceptance_cmap),
        ax=ax,
        pad=0.02,
    )
    colorbar.set_label("Acceptance Rate (%)")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    _add_cost_tradeoff_legends(
        ax,
        colorbar.ax,
        lambda_values,
        lambda_colors,
        mode_values,
        fleet_values,
        fleet_markers,
    )

    ax.set_xlabel("Operator Unit Cost (€/passenger)")
    ax.set_ylabel("Passenger Unit Cost (€/passenger)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    plt.savefig(outpng, dpi=600, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return outpng


def _valid_cost_tradeoff_frame(
    frame: pd.DataFrame,
    x_key: str,
    y_key: str,
    color_key: str,
) -> pd.DataFrame:
    required_columns = ["lambda", "mode_id", "fleet_size", x_key, y_key, color_key]
    missing_columns = set(required_columns) - set(frame.columns)
    if missing_columns:
        raise KeyError(f"Missing cost tradeoff columns: {sorted(missing_columns)}")

    valid_frame = frame.loc[:, required_columns].copy()
    for column in required_columns:
        valid_frame[column] = pd.to_numeric(valid_frame[column], errors="coerce")

    valid_frame = valid_frame.dropna(subset=required_columns)
    valid_frame["mode_id"] = valid_frame["mode_id"].astype(int)
    valid_frame["fleet_size"] = valid_frame["fleet_size"].astype(int)
    valid_frame["lambda"] = valid_frame["lambda"].astype(float)
    return valid_frame


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
        candidate = float(value) if column == "lambda" else int(value)
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


def _add_cost_tradeoff_legends(
    ax,
    colorbar_ax,
    lambda_values: Sequence,
    lambda_colors: dict,
    mode_values: Sequence[int],
    fleet_values: Sequence[int],
    fleet_markers: dict[int, str],
) -> None:
    lambda_handles = [
        Line2D(
            [],
            [],
            color=lambda_colors[lambda_value],
            linestyle="-",
            linewidth=2.0,
            label=f"λ = {_format_number(lambda_value)} requests/h",
        )
        for lambda_value in lambda_values
    ]
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

    colorbar_box = colorbar_ax.get_position()
    lambda_legend = ax.figure.legend(
        handles=lambda_handles,
        title="lambda",
        loc="upper left",
        bbox_to_anchor=(colorbar_box.x1 + 0.02, colorbar_box.y1),
        bbox_transform=ax.figure.transFigure,
        borderaxespad=0.0,
        framealpha=0.88,
        fontsize=8,
    )

    mode_legend = ax.legend(
        handles=mode_handles,
        title="mode_id",
        loc="lower left",
        framealpha=0.88,
        fontsize=8,
    )
    ax.add_artist(mode_legend)

    ax.legend(
        handles=fleet_handles,
        title="fleet_size",
        loc="upper right",
        framealpha=0.88,
        fontsize=8,
    )


def _format_number(value) -> str:
    numeric_value = float(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:g}"


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

    for group_key, group_records in grouped_records.items():
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
        draw_confidence_bands(ax, frame, x_key, y_key, confidence=confidence)
    if show_ols:
        draw_ols_lines(ax, frame, x_key, y_key, confidence=confidence)
    if show_lowess:
        draw_lowess_lines(ax, frame, x_key, y_key)


def fit_ols_by_mode(
    frame: pd.DataFrame,
    x_key,
    y_key,
    confidence: float = 0.95,
) -> dict[int, dict]:
    valid_frame = _valid_regression_frame(frame, x_key, y_key)
    mode_ids = sorted(set(MODE_LABELS) | set(valid_frame["mode_id"].astype(int)))
    results: dict[int, dict] = {}

    for mode_id in mode_ids:
        mode_frame = valid_frame[valid_frame["mode_id"].astype(int) == mode_id].copy()
        x_values = mode_frame[x_key].astype(float)
        y_values = mode_frame[y_key].astype(float)
        n = int(len(mode_frame))
        x_min = float(x_values.min()) if n else None
        x_max = float(x_values.max()) if n else None

        base_result = {
            "mode_id": mode_id,
            "mode_label": MODE_LABELS.get(mode_id, f"Mode {mode_id}"),
            "n": n,
            "x_min": x_min,
            "x_max": x_max,
            "confidence_level": confidence,
        }

        if n < 2 or x_values.nunique() < 2:
            results[mode_id] = {
                **base_result,
                "status": "insufficient_data",
                "model": None,
                "x_grid": np.array([], dtype=float),
                "prediction": pd.DataFrame(),
                "slope": None,
                "intercept": None,
                "r_squared": None,
                "p_value": None,
            }
            continue

        x_series = pd.Series(x_values.to_numpy(dtype=float), name=str(x_key))
        y_series = pd.Series(y_values.to_numpy(dtype=float), name=str(y_key))
        design = sm.add_constant(x_series, has_constant="add")
        model = sm.OLS(y_series, design).fit()

        x_grid = np.linspace(float(x_values.min()), float(x_values.max()), 100)
        prediction_design = sm.add_constant(
            pd.Series(x_grid, name=str(x_key)),
            has_constant="add",
        )
        prediction = model.get_prediction(prediction_design).summary_frame(
            alpha=1.0 - confidence,
        )

        results[mode_id] = {
            **base_result,
            "status": "ok",
            "model": model,
            "x_grid": x_grid,
            "prediction": prediction,
            "slope": float(model.params[str(x_key)]),
            "intercept": float(model.params["const"]),
            "r_squared": float(model.rsquared),
            "p_value": float(model.pvalues[str(x_key)]),
        }

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


def draw_confidence_bands(
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
    valid_frame = _valid_regression_frame(frame, x_key, y_key)
    for mode_id, mode_frame in valid_frame.groupby("mode_id"):
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


def build_regression_report(
    frame: pd.DataFrame,
    x_key,
    y_key,
    plot_name: str,
    panel: str,
    confidence: float = 0.95,
) -> pd.DataFrame:
    rows = []
    for result in fit_ols_by_mode(frame, x_key, y_key, confidence).values():
        rows.append(
            {
                "plot_name": plot_name,
                "panel": panel,
                "x_key": x_key,
                "y_key": y_key,
                "mode_id": result["mode_id"],
                "mode_label": result["mode_label"],
                "n": result["n"],
                "slope": result["slope"],
                "intercept": result["intercept"],
                "r_squared": result["r_squared"],
                "p_value": result["p_value"],
                "confidence_level": result["confidence_level"],
                "x_min": result["x_min"],
                "x_max": result["x_max"],
                "status": result["status"],
            }
        )

    return pd.DataFrame(rows, columns=REGRESSION_REPORT_COLUMNS)


def export_regression_report(
    report_frames,
    output_dir: Path,
    timestamp: str | None = None,
) -> Path:
    if isinstance(report_frames, pd.DataFrame):
        frames = [report_frames]
    else:
        frames = list(report_frames)

    if frames:
        report_frame = pd.concat(frames, ignore_index=True)
    else:
        report_frame = pd.DataFrame(columns=REGRESSION_REPORT_COLUMNS)

    output_path = Path(output_dir) / f"regression_{_regression_timestamp(timestamp)}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing_frame = pd.read_csv(output_path)
        report_frame = pd.concat([existing_frame, report_frame], ignore_index=True)

    report_frame = report_frame.reindex(columns=REGRESSION_REPORT_COLUMNS)
    report_frame.to_csv(output_path, index=False)
    return output_path


def _valid_regression_frame(frame: pd.DataFrame, x_key, y_key) -> pd.DataFrame:
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
    return valid_frame


def _regression_timestamp(timestamp: str | None) -> str:
    global _DEFAULT_REGRESSION_TIMESTAMP
    if timestamp is not None:
        return timestamp
    if _DEFAULT_REGRESSION_TIMESTAMP is None:
        _DEFAULT_REGRESSION_TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M%S")
    return _DEFAULT_REGRESSION_TIMESTAMP


# 2x2 subplots
def plts_4s(path_file: Path,
            x_key, y_key, z_key) -> Path:
    # ===== 1. 读取 JSON 文件 =====
    records, json_path = load_records(Path(path_file))
    outpng = Path(json_path).with_name("4s_" + json_path.stem).with_suffix(".png")



    # ===== 3. 创建 2x2 图 =====
    fig = plt.figure(figsize=(14, 10))
    ax_3d = fig.add_subplot(221, projection="3d")
    ax_xy = fig.add_subplot(222)
    ax_xz = fig.add_subplot(223)
    ax_yz = fig.add_subplot(224)

    ax_3d = model_3d(ax_3d, records, x_key, y_key, z_key)
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
    plt.show()
    # plt.pause(0.1)  # 确保图像显示出来
    #plt.close(fig)
    return outpng

# 3d scatter plot
def model_3d(ax, records, x_key, y_key, z_key):
    # 为避免同一场景的点完全重叠，给不同 mode 一个很小的偏移
    offsets = {
        1: (-0.03, -0.03, -0.8),
        2: (-0.03,  0.03, -0.3),
        3: ( 0.03, -0.03,  0.3),
        4: ( 0.03,  0.03,  0.8),
    }
    add_scatter(ax, records, 
                x_key, y_key, z_key,
                offsets,
                )

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_zlabel(z_key)
    ax.set_title("Optimal Modes Across Selected Scenarios")
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


def add_scatter(ax, records: list[dict],
                x_key, y_key, z_key,
                offsets: dict[int, tuple[float, float, float]] | None = None,                                   
                ) -> None:
    if offsets is None:
        offsets = {}
    added_labels: set[int] = set()

    for row in records:
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

def add_projections(ax, records: list[dict], 
                    x_key = None, y_key = None, z_key = None, 
                    plane = None,
                    offsets: dict[int, tuple[float, float, float]] | None = None,
                    ) -> None:
    if offsets is None:
        offsets = {}
    for row in records:
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

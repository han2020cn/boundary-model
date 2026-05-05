from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
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
def plts_2d(frame: pd.DataFrame, output_path: Path, x_key, y_key, types: list[str]) -> Path:
    outpng = Path(output_path)/"2d_plots.png"
    outpng.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    _draw_2d_scatter(ax, frame, x_key, y_key, types, "x-y scatter plot", show_legend=True)

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
) -> Path:
    outpng = Path(output_dir) / f"2d_pairs_{datetime.now().strftime('%m%d_%H%M%S')}.png"
    outpng.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False, sharey=False)
    _draw_2d_scatter(axes[0], left_frame, x_key, y_key, types, left_title, show_legend=False)
    _draw_2d_scatter(axes[1], right_frame, x1_key, y1_key, types, right_title, show_legend=False)
    _draw_unserved_portion_lines(axes[1], right_frame, x1_key)

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


def _draw_unserved_portion_lines(ax, frame: pd.DataFrame, x_key) -> None:
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
            total_requests = float(row["total_requests"])
            unserved_requests = float(row["unserved_requests"])
            portion = 0.0 if total_requests == 0.0 else unserved_requests / total_requests
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

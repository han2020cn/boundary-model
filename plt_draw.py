from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# ===== 2. 为不同 mode_id 设置颜色 =====
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

# results.json xyz分别是net_expenditure, avg_service_time, total_requests
def results_3d(path_file: Path) -> Path:
    records, json_path = load_records(Path(path_file))
    outpng = json_path.with_suffix(".png")
    x_key = "net_expenditure"
    y_key = "avg_service_time"
    z_key = "total_requests"
    x_values = [float(row[x_key]) for row in records]
    y_values = [float(row[y_key]) for row in records]
    z_values = [float(row[z_key]) for row in records]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # add_surface(ax, x_values, y_values, z_values)
    add_scatter(ax, records, 
                x_key, y_key, z_key,
                )

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_zlabel(z_key)
    ax.set_title("Optimal Modes Across Selected Scenarios")
    ax.legend(title="optimal mode")

    plt.tight_layout()
    plt.savefig(outpng, dpi=600, bbox_inches="tight")
    plt.show()
    #plt.close(fig)

    return outpng


# optimals.json xyz分别是ht, hs, lambda
def optimals_3d(path_file: Path):
    # ===== 1. 读取 JSON 文件 =====
    records, json_path = load_records(Path(path_file))
    outpng = Path(json_path).with_suffix(".png")
    x_key = "ht"
    y_key = "hs"
    z_key = "lambda"
    # 为避免同一场景的点完全重叠，给不同 mode 一个很小的偏移
    offsets = {
        1: (-0.03, -0.03, -0.8),
        2: (-0.03,  0.03, -0.3),
        3: ( 0.03, -0.03,  0.3),
        4: ( 0.03,  0.03,  0.8),
    }

    # ===== 3. 创建 2x2 图 =====
    fig = plt.figure(figsize=(14, 10))
    ax_3d = fig.add_subplot(221, projection="3d")
    ax_xy = fig.add_subplot(222)
    ax_xz = fig.add_subplot(223)
    ax_yz = fig.add_subplot(224)

    ax_3d = add_scatter(ax_3d, records, 
                x_key, y_key, z_key,
                offsets,
                )
    ax_xy = add_projections(ax_xy, records, 
                 x_key, y_key, z_key,
                 'xy',
                 offsets,
                 )
    ax_xz = add_projections(ax_xz, records, 
                 x_key, y_key, z_key,
                 'xz',
                 offsets,
                 )
    ax_yz = add_projections(ax_yz, records, 
                 x_key, y_key, z_key,
                 'yz',
                 offsets,
                 )    

    # ===== 4. 坐标轴和标题 =====
    x_min, x_max = -0.08, 1.05
    y_min, y_max = -0.08, 1.05
    z_min, z_max = 18, 62

    # 3D
    ax_3d.set_xlim(x_min, x_max)
    ax_3d.set_ylim(y_min, y_max)
    ax_3d.set_zlim(z_min, z_max)
    ax_3d.set_xticks([0.0, 0.5, 1.0])
    ax_3d.set_yticks([0.0, 0.5, 1.0])
    ax_3d.set_zticks([20, 40, 60])
    ax_3d.set_xlabel("temporal")
    ax_3d.set_ylabel("spatial")
    ax_3d.set_zlabel("lambda")
    ax_3d.set_title("3D Scatter Plot of Scenarios by Mode ID")
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
    fig.legend(handles, labels, title="mode_id", loc="upper center", ncol=4)

    plt.tight_layout()

    plt.savefig(outpng, dpi=600, bbox_inches="tight")
    plt.show(block = False)
    plt.pause(0.1)  # 确保图像显示出来
    #plt.close(fig)



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



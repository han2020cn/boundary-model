from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# from main import main


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


def load_records() -> tuple[list[dict], Path]:
    results_path = Path("/home/han/from-codex/boundary-model/rs/scenario_results_20260413_174928.json")
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


def add_scatter(ax, records: list[dict]) -> None:
    added_labels: set[int] = set()

    for row in records:
        mode_id = int(row["mode_id"])
        label = MODE_LABELS[mode_id] if mode_id not in added_labels else None
        if label is not None:
            added_labels.add(mode_id)

        ax.scatter(
            float(row["net_expenditure"]),
            float(row["avg_service_time"]),
            float(row["total_requests"]),
            color=MODE_COLORS.get(mode_id, "black"),
            edgecolors="black",
            linewidths=0.4,
            s=60,
            depthshade=False,
            label=label,
        )


def plot_optimal_modes() -> Path:
    records, json_path = load_records()
    outpng = json_path.with_suffix(".png")

    x_values = [float(row["net_expenditure"]) for row in records]
    y_values = [float(row["avg_service_time"]) for row in records]
    z_values = [float(row["total_requests"]) for row in records]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # add_surface(ax, x_values, y_values, z_values)
    add_scatter(ax, records)

    ax.set_xlabel("net_expenditure")
    ax.set_ylabel("avg_service_time")
    ax.set_zlabel("total_requests")
    ax.set_title("Optimal Modes Across Selected Scenarios")
    ax.legend(title="optimal mode")

    plt.tight_layout()
    plt.savefig(outpng, dpi=600, bbox_inches="tight")
    plt.show()
    #plt.close(fig)

    return outpng


if __name__ == "__main__":
    output_path = plot_optimal_modes()
    print(f"Saved 3D optimal-mode plot to {output_path}")

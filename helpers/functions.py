import json
import os
from pathlib import Path
import tempfile
import pandas as pd
from datetime import date, datetime
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

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


# json to excel: convert files
def transfer_json_to_excel(file_path: Path) -> Path:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    output_path = file_path.with_suffix(".xlsx")
    df.to_excel(output_path, index=False)
    return output_path


def write_json(data, output_path: Path) -> Path:
    """Atomically write JSON data, creating the parent directory as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def export_json(frame: pd.DataFrame, output_path: Path) -> Path:
    json_records = json.loads(frame.to_json(orient="records"))
    return write_json(json_records, output_path)


def extend_json(frame: pd.DataFrame, output_path: Path,
                        ) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    else:
        records = []

    records.extend(json.loads(frame.to_json(orient="records")))
    return write_json(records, output_path)


MODE_LABELS = {
    1: "Mode 1",
    2: "Mode 2",
    3: "Mode 3",
    4: "Mode 4",
}


def _regression_report(
    frame: pd.DataFrame,
    output_dir: Path,
    x_key,
    y_key,
    plot_name: str,
    panel: str | None = None,
    confidence: float = 0.95,
    date: str | None = None,
) -> Path:
    required_columns = {"mode_id", x_key, y_key}
    valid_frame = frame.loc[:, ["mode_id", x_key, y_key]].copy()

    valid_frame[x_key] = pd.to_numeric(valid_frame[x_key], errors="coerce")
    valid_frame[y_key] = pd.to_numeric(valid_frame[y_key], errors="coerce")
    valid_frame["mode_id"] = pd.to_numeric(valid_frame["mode_id"], errors="coerce")

    valid_frame = valid_frame.dropna(subset=["mode_id", x_key, y_key])
    valid_frame["mode_id"] = valid_frame["mode_id"].astype(int)
    

    mode_ids = sorted(set(MODE_LABELS) | set(valid_frame["mode_id"]))
    rows = []

    for mode_id in mode_ids:
        mode_frame = valid_frame[valid_frame["mode_id"] == mode_id].copy()
        x_values = mode_frame[x_key].astype(float)
        y_values = mode_frame[y_key].astype(float)
        n = int(len(mode_frame))
        x_min = float(x_values.min()) if n else None
        x_max = float(x_values.max()) if n else None

        result = {
            "plot_name": plot_name,
            "panel": panel,
            "x_key": x_key,
            "y_key": y_key,
            "mode_id": mode_id,
            "mode_label": MODE_LABELS.get(mode_id, f"Mode {mode_id}"),
            "n": n,
            "slope": None,
            "intercept": None,
            "r_squared": None,
            "p_value": None,
            "confidence_level": confidence,
            "x_min": x_min,
            "x_max": x_max,
            "status": "insufficient_data",
        }

        if n >= 2 and x_values.nunique() >= 2:
            x_series = pd.Series(x_values.to_numpy(dtype=float), name=str(x_key))
            y_series = pd.Series(y_values.to_numpy(dtype=float), name=str(y_key))
            design = sm.add_constant(x_series, has_constant="add")
            model = sm.OLS(y_series, design).fit()

            result.update(
                {
                    "slope": float(model.params[str(x_key)]),
                    "intercept": float(model.params["const"]),
                    "r_squared": float(model.rsquared),
                    "p_value": float(model.pvalues[str(x_key)]),
                    "status": "ok",
                }
            )

        rows.append(
            {column: result[column] for column in REGRESSION_REPORT_COLUMNS}
        )

    report_frame = pd.DataFrame(rows, columns=REGRESSION_REPORT_COLUMNS)

    output_path = Path(output_dir) / f"regression_{date}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing_frame = pd.read_csv(output_path)
        report_frame = pd.concat([existing_frame, report_frame], ignore_index=True)

    report_frame = report_frame.reindex(columns=REGRESSION_REPORT_COLUMNS)
    report_frame.to_csv(output_path, index=False)
    return output_path


def draw_loops(loops, output_dir: Path | None = None) -> Path: # 绘制所有路线的图
    loop_items = tuple(loops) if isinstance(loops, (list, tuple)) else (loops,)
    if not loop_items:
        raise ValueError("draw_loops requires at least one loop")

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "loops.png"
    fig, ax = plt.subplots(figsize=(8, 8))

    for index, loop in enumerate(loop_items):
        stops = tuple(loop.fixed_stop_indices.keys())
        if len(stops) < 2:
            continue
        closed_stops = stops + (stops[0],)
        xs, ys = zip(*(_node_xy(stop) for stop in closed_stops))
        ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.0)

    ax.set_title("Loop Routes")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    # ax.legend(loc="best", framealpha=0.88)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    # plt.close(fig)
    return png_path


def _node_xy(node) -> tuple[float, float]:  # 将节点转换为(x, y)坐标，如果节点是一个包含两个数值的元组，则返回这些数值作为坐标，否则抛出ValueError异常
    if (
        isinstance(node, tuple)
        and len(node) == 2
        and all(isinstance(value, (int, float)) for value in node)
    ):
        return float(node[0]), float(node[1])
    raise ValueError(f"loop stop must be a numeric (x, y) tuple: {node!r}")

        

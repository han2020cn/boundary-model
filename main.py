from __future__ import annotations

from datetime import datetime
from itertools import product
import json
from pathlib import Path
import pandas as pd
import random
###
import json
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

from demand_generation import generate_requests
from mode_set import (
    RESULT_COLUMNS, # column names for the results DataFrame
    build_grid_graph,
    evaluate_mode_1,
    evaluate_mode_2,
    evaluate_mode_3,
    evaluate_mode_4,
)

BASE_SEED = 20260402
GRID_SIZE = 9
HORIZON = 180 # 时间范围 / 仿真时域（time horizon / simulation horizon）
LAMBDA_LEVELS = (20, 40, 60) # 需求强度或到达率（demand intensity / arrival rate）
HS_LEVELS = (0.0, 0.5, 1.0) # 空间异质性（spatial heterogeneity）
HT_LEVELS = (0.0, 0.5, 1.0) # 时间异质性（temporal heterogeneity）
# scenarios_num= len(LAMBDA_LEVELS) * len(HS_LEVELS) * len(HT_LEVELS)

def build_scenario_frame(run_seed: int | None = None) -> pd.DataFrame:
    if run_seed is None:
        run_seed = random.randint(0, 10**9)

    rows = []
    for index, (lambda_value, hs, ht) in enumerate(
        product(LAMBDA_LEVELS, HS_LEVELS, HT_LEVELS),
        start=1,
    ):
        rows.append(
            {
                "scenario_id": f"S{index:02d}_l{lambda_value}_hs{hs:.1f}_ht{ht:.1f}",
                "lambda": int(lambda_value),
                "hs": float(hs),
                "ht": float(ht),
                "seed": run_seed + index - 1,
            }
        )
    return pd.DataFrame(rows)


def run_scenarios() -> pd.DataFrame:
    graph = build_grid_graph(GRID_SIZE)
    scenario_frame = build_scenario_frame()
    result_rows = []

    for scenario in scenario_frame.to_dict(orient="records"):
        requests = generate_requests(
            lambda_value=float(scenario["lambda"]),
            hs=float(scenario["hs"]),
            ht=float(scenario["ht"]),
            seed=int(scenario["seed"]),
            grid_size=GRID_SIZE,
            horizon=HORIZON,
        )

        mode_1_result = evaluate_mode_1(requests, scenario, graph)
        result_rows.append(mode_1_result)

        benchmark_expenditure = (
            float(mode_1_result["net_expenditure"])
            if mode_1_result["feasible"]
            else None
        )
        result_rows.append(
            evaluate_mode_2(requests, scenario, graph, benchmark_expenditure)
        )
        result_rows.append(
            evaluate_mode_3(requests, scenario, graph, benchmark_expenditure)
        )
        result_rows.append(
            evaluate_mode_4(requests, scenario, graph, benchmark_expenditure)
        )

    return pd.DataFrame(result_rows, columns=RESULT_COLUMNS)


def export_results(results_frame: pd.DataFrame, output_dir: Path) ->  Path:
    #output_dir.mkdir(parents=True, exist_ok=True)
    #csv_path = output_dir / "scenario_results.csv"
    #results_frame.to_csv(csv_path, index=False)
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"scenario_results_{date}.json"
    json_records = json.loads(results_frame.to_json(orient="records"))
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_records, handle, ensure_ascii=False, indent=2)

    return json_path


def main() -> pd.DataFrame:
    results_frame = run_scenarios()
    output_dir  = Path("/home/han/from-codex/boundary-model/rs")
    json_path = export_results(results_frame, output_dir)
    return json_path

'''
选出每个 scenario 的 optimal
'''
def optimal_modes(json_path: Path):
    # 读取结果
    df = pd.read_json(json_path)
    
    # 只保留可行方案
    feasible_df = df[df["feasible"] == True].copy()

    # 先按 scenario_id 分组，再按 avg_service_time、net_expenditure 排序
    feasible_df = feasible_df.sort_values(
        by=["scenario_id", "avg_service_time", "net_expenditure"]
    )

    # 每个 scenario 取第一条 = optimal mode
    optimal_df = feasible_df.groupby("scenario_id", as_index=False).first()
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonpath_opt = json_path.with_name(f"optimals_{date}.json")
    optimal_df.to_json(jsonpath_opt, orient="records", force_ascii=False, indent=2)
    return str(jsonpath_opt)



#draw 3D scatter plot of optimal modes

def draw_3d(json_path: Path):
    # ===== 1. 读取 JSON 文件 =====
    jsonpath = optimal_modes(json_path)  
    outpng = Path(jsonpath).with_suffix(".png")

    with open(jsonpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ===== 2. 为不同 mode_id 设置颜色 =====
    mode_colors = {
        1: "tab:grey", # 1 是基准方案(固定路线模式)，颜色较淡
        2: "tab:blue", # 2 是方案二deviated，颜色较亮
        3: "tab:orange", # 3 是方案三DRT，颜色较亮
        4: "tab:red", # 4 是方案四hub-and-spoke，颜色较亮
    }

    mode_labels = {
        1: "Mode 1",
        2: "Mode 2",
        3: "Mode 3",
        4: "Mode 4",
    }

    # 为避免同一场景的四个点完全重叠，给不同 mode 一个很小的偏移
    offsets = {
        1: (-0.03, -0.03, -0.8),
        2: (-0.03,  0.03, -0.3),
        3: ( 0.03, -0.03,  0.3),
        4: ( 0.03,  0.03,  0.8),
    }

    # ===== 3. 创建 3D 图 =====

    fig = plt.figure(figsize=(10, 8)) 
    ax = fig.add_subplot(111, projection="3d") # 为了避免 legend 重复，只给每个 mode_id 添加一次标签 
    added_label = set() 
    for row in data: 
        x = row["ht"] 
        y = row["hs"] 
        z = row["lambda"] 
        mode_id = row["mode_id"] 
        dx, dy, dz = offsets.get(mode_id, (0, 0, 0)) 
        label = mode_labels[mode_id] if mode_id not in added_label else None 
        if label is not None: 
            added_label.add(mode_id) 
        ax.scatter( x + dx, y + dy, z + dz, color=mode_colors.get(mode_id, "black"), s=60, alpha=0.85, label=label ) 

    # ===== 4. 坐标轴和标题 =====
    
    ax.set_xlabel("ht") 
    ax.set_ylabel("hs") 
    ax.set_zlabel("lambda") 
    ax.set_title("3D Scatter Plot of Scenarios by Mode ID")

    # 设置刻度，更清楚
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_zticks([20, 40, 60])

    #
    # ===== 5. 图例 =====
    ax.legend(title="mode_id")

    plt.tight_layout()

    plt.savefig(outpng, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    json_path = main()
    print(f"JSON path: {json_path}")
    optimal_modes(json_path)
    draw_3d(json_path)

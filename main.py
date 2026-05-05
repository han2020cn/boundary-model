from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
import scenarios as sc
import plt_draw as plt


BASE_SEED = 20260402
GRID_SIZE = 10
HORIZON = 180 # 时间范围 / 仿真时域（time horizon / simulation horizon）
LAMBDA_LEVELS = tuple(range(1, 101, 5)) # 需求强度或到达率（demand intensity / arrival rate）
HS_LEVELS = (0.5,) # 空间异质性（spatial heterogeneity）
HT_LEVELS = (0.5,) # 时间异质性（temporal heterog.eneity）tuple(i/10 for i in range(0, 11))
SERVICE_POLICY = "strict"
PREBOOKING_ALPHA = 0.5 # prebooking rate
REPLICATION = True #是否复现
# scenarios_num= len(LAMBDA_LEVELS) * len(HS_LEVELS) * len(HT_LEVELS)


def extend_json_records(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    else:
        records = []

    records.extend(json.loads(frame.to_json(orient="records")))
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    return output_path



def main(scene: int, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    if scene == 1:
        requests, results_frame = sc.demand_scenarios(GRID_SIZE, HORIZON, 
                                     LAMBDA_LEVELS, HS_LEVELS, HT_LEVELS,
                                     BASE_SEED,
                                     output_dir,
                                     replication=REPLICATION,
                                     service_policy=SERVICE_POLICY,
                                     prebooking_alpha=PREBOOKING_ALPHA,
                                     )
        sc_type = "demand"

    if scene == 2:
        lambda_, hs_, ht_ = 40, 0.5, 0.5
        fleet_sizes = (3, 6, 9, 12, 15)
        capacities = (15, 30, 45)
        seed_count = 5
        results_frame = sc.cost_scenarios(lambda_, hs_, ht_, 
                                     GRID_SIZE, HORIZON, 
                                     fleet_sizes, capacities, 
                                     seed_count, BASE_SEED,
                                     service_policy=SERVICE_POLICY,
                                     prebooking_alpha=PREBOOKING_ALPHA,
                                     )
        sc_type = "cost"
    
    optimals_frame = sc.optimals(results_frame)

    if REPLICATION:
        results_path = extend_json_records(results_frame, output_dir / "com_results.json")
        optimals_path = extend_json_records(optimals_frame, output_dir / "com_optimals.json")
    else:
        results_path = sc.export_files(results_frame, output_dir, sc_type, "rs")
        optimals_path = sc.export_files(optimals_frame, output_dir, sc_type, "ops")
    # print(f"JSON path: {optimals_path}")

    return results_frame, optimals_frame, results_path, optimals_path

# json to excel: convert files
def json_to_excel(file_path: Path) -> Path:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    output_path = file_path.with_suffix(".xlsx")
    df.to_excel(output_path, index=False)
    

if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "rs"
    results_frame, optimals_frame, rs_path, optimals_path = main(1, output_dir)

    x = "served_requests"
    x1 = "served_requests"
    y = "net_expenditure"
    y1 = "avg_service_time"
    z = "lambda"
    types = ["mode_id"]
    plt.plts_2d(results_frame,output_dir,y,y1,types)
    plt.plts_2d_pair(
        results_frame,
        results_frame,
        output_dir,
        x,
        x1,
        y,
        y1,
        types,
        left_title=y,
        right_title=y1,
        prebooking_alpha=PREBOOKING_ALPHA,
    ) #画图
    #plts_3d xyz图, plts_2d xy图, plts_4s 2x2图

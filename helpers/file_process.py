import json
from pathlib import Path
import pandas as pd
import scenarios as sc

def merge_file(input_files, dir_path, suffix, init_filename,merged_filename, optimal_filename):

    merged = []

    for filename in input_files:
        path = dir_path / filename

        if not path.is_file():
            raise FileNotFoundError(f"找不到文件：{path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise TypeError(f"{path} 的 JSON 顶层必须是数组")
        
        feasible_results = [
            row
            for row in data
            if row["feasible"] == True
        ]
        merged.extend(feasible_results)

    print(
        f"已合并 {len(input_files)} 个文件，"
        f"共 {len(merged)} 条记录"
    )

    output_path_init = dir_path / f"{init_filename}_{suffix}.json"
    with output_path_init.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    #calculate mean of each scenario_id
    merged_df = pd.DataFrame(merged)
    merged_df = sc.mean_result(merged_df) # feasible results only then mean
    output_path_mean = dir_path / f"{merged_filename}_{suffix}.json"

    #选出每个 scenario 的 optimal
    with output_path_mean.open("w", encoding="utf-8") as f:
        json.dump(merged_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    min_objective = merged_df.groupby("scenario_id")["objective_value"].transform("min")
    optimal_df = merged_df[merged_df["objective_value"] == min_objective].copy()

    optimal_path = dir_path / f"{optimal_filename}_{suffix}.json"

    with optimal_path.open("w", encoding="utf-8") as f:
        json.dump(optimal_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

   

def filter_object(results_frame, column, value):
    """排除指定列等于指定值的行，并返回新的 DataFrame。"""
    return results_frame.loc[results_frame[column] != value].copy()
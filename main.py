from __future__ import annotations

import pandas as pd
from datetime import datetime

from pathlib import Path
import scenarios as sc
import demand_generation as dg
import functions as fs
#导入class
from config import config, nets, fleet


output_dir = config.output_dir
date = config.date
scene = config.scene

def main(config, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_frame = None
    results = output_dir / "de_result_260601_1214.json"  
    if scene ==  "de":
        requests, results_frame = sc.demand_scenarios(config, nets, fleet, output_dir,          
                                     )
    if scene ==  "co":
        requests, results_frame = sc.cost_scenarios(config, nets, fleet, output_dir,
                                     )
        
    if results_frame is not None:
        results = results_frame      
    
    optimals_frame = sc.optimals(results)

    fs.export_json(optimals_frame, output_dir / f"{scene}_optimal_{date}.json")

    return results_frame, optimals_frame




if __name__ == "__main__":
        
    results_frame, optimals_frame = main(config, output_dir)
    # tem_dir = Path(__file__).resolve().parent /"rs" 
    # results_frame = pd.read_json(tem_dir / "de_result_260528_1623.json")
    
    ##画出请求的起点和终点分布图
    # plt._draw_request(f"requests_{date}.json", nets, output_dir/"requests")     #requests文件名，保存路径
    #f"requests_{date}.json"

    # print("Processing complete.")
    # input("Press Enter to continue...")
    # print("Continuing program...")

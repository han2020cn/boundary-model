from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

import scenarios as sc
import helpers.functions as fs
#导入class
from helpers.config import Config, Grid, Radial, Fleet

# 场景选择：1-需求场景，2-成本场景 lambdas = tuple(range(10, 60, 20))
config = Config(
    lambdas=tuple(range(90,100, 10)),
    hs=(0.8,),
    ht=(0.8,),
    replication=False,
    sc=1,
    seed_count=1,
)
nets = Grid(_type = 'grid', grid = 50, grid_len = 100, num_routes = 2)
# nets = Radial(_type = "hub_spoke", spoke_count = 8, ring_radial = (5, 10, 15))
fleet = Fleet(cap = 30)

def main(
    config,
    nets,
    fleet,
    *,
    shard_id: int = 0,
    shard_count: int = 1,
    run_id: str | None = None,
    output_root: Path | None = None,
    artifact_policy: str = "all",
    resume: bool = False,
) -> pd.DataFrame:
    if config.scene == "de":
        return sc.demand_scenario(
            config,
            nets,
            fleet,
            shard_id=shard_id,
            shard_count=shard_count,
            run_id=run_id,
            output_root=output_root,
            artifact_policy=artifact_policy,
            resume=resume,
        )
    if config.scene == "co":
        if (
            shard_id != 0
            or shard_count != 1
            or run_id is not None
            or output_root is not None
            or resume
            or artifact_policy != "all"
        ):
            raise NotImplementedError(
                "HPC sharding currently supports demand scenarios (sc=1) only"
            )
        return sc.cost_scenario(config, nets, fleet)
    raise ValueError(f"unsupported scenario selection: sc={config.sc!r}")


def _environment_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must contain an integer") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run boundary-model demand simulations locally or as an HPC shard",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=_environment_int("SLURM_ARRAY_TASK_ID", 0),
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=_environment_int("SLURM_ARRAY_TASK_COUNT", 1),
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("SLURM_ARRAY_JOB_ID"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--artifact-policy",
        choices=sorted(sc.ARTIFACT_POLICIES),
        default="all",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a compatible shard checkpoint instead of refusing overwrite",
    )
    return parser

def _local_result() -> pd.DataFrame:
    file_name = "de_result_260602_1927.json"
    results_path = config.output_dir / file_name
    sc, result_type, config.date, time = file_name.removesuffix(".json").split("_")
    optimals_results = sc.optimals(results_path)
    optimals_path = config.output_dir / f"{sc}_optimal_{config.date}_{time}.json"
    fs.export_json(optimals_results, optimals_path)


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(
        config,
        nets,
        fleet,
        shard_id=args.shard_id,
        shard_count=args.shard_count,
        run_id=args.run_id,
        output_root=args.output_root,
        artifact_policy=args.artifact_policy,
        resume=args.resume,
    )

    
    # print("Processing complete.")
    # input("Press Enter to continue...")
    # print("Continuing program...")

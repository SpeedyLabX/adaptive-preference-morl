#!/usr/bin/env python3
"""Generate hei-research .toml recipes for the 21 Stage-1A vertex-specialist anchors and 50 GPU arm runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "protocols" / "stage1_simplex_face_exposure_v1.json"
RECIPE_DIR = REPO_ROOT / "configs" / "recipes" / "stage1a"
JOB_SCRIPT = REPO_ROOT / "scripts" / "run_stage1a_gpu_job.py"
MINECART_STANDALONE_SCRIPT = REPO_ROOT / "scripts" / "run_stage1a_minecart_kaggle.py"


def write_recipe(
    path: Path,
    name: str,
    dataset: str,
    seed: int,
    est_seconds: int,
    max_seconds: int,
    script_path: Path = JOB_SCRIPT,
) -> None:
    rel_script = Path(os.path.relpath(script_path, path.parent)).as_posix()
    content = f"""[experiment]
name = "{name}"
script = "{rel_script}"
seed = {seed}
device = "cuda"
min_vram_mib = 512
estimated_seconds = {est_seconds}
max_seconds = {max_seconds}
dataset = "{dataset}"
dataset_mib = 0
compatibility = "stage1a-simplex-ppo-v1"
data_class = "synthetic"
epochs = 1

[local]
conda_env = "morl-py311"

[colab]
gpu = "T4"
units_per_hour = 2.0
max_units = 6.0

[kaggle]
accelerator = "NvidiaTeslaT4"
dataset_sources = []
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    benchmarks = {b["env_id"]: b for b in protocol["benchmarks"]}

    anchor_count = 0
    arm_count = 0
    for env_id in ("minecart-v0", "mo-lunar-lander-v3"):
        b = benchmarks[env_id]
        est_s = 6700 if env_id == "minecart-v0" else 3500
        max_s = 9000 if env_id == "minecart-v0" else 5400
        short_env = "minecart" if env_id == "minecart-v0" else "lunar"

        # 21 Vertex-specialist anchor recipes (9 Minecart standalone Kaggle/local + 12 Lunar local)
        anchor_script = MINECART_STANDALONE_SCRIPT if env_id == "minecart-v0" else JOB_SCRIPT
        anchor_max_s = 14400 if env_id == "minecart-v0" else max_s
        for v in range(int(b["anchor_spec"]["vertices"])):
            for seed in b["anchor_spec"]["anchor_seeds"]:
                name = f"stage1a-{short_env}-anchor-v{v}-s{seed}"
                write_recipe(
                    RECIPE_DIR / "anchors" / f"{name}.toml",
                    name=name,
                    dataset=f"{env_id}|anchor|vertex={v}",
                    seed=int(seed),
                    est_seconds=est_s,
                    max_seconds=anchor_max_s,
                    script_path=anchor_script,
                )
                anchor_count += 1

        # 60 Stage-1A arm recipes (30 Minecart + 30 Lunar under protocol v1.3)
        for arm in b["arms_included"]:
            for seed in b["paired_seeds"]:
                name = f"stage1a-{short_env}-{arm.replace('_', '-')}-s{seed}"
                write_recipe(
                    RECIPE_DIR / "arms" / short_env / f"{name}.toml",
                    name=name,
                    dataset=f"{env_id}|arm|{arm}",
                    seed=int(seed),
                    est_seconds=est_s,
                    max_seconds=max_s,
                    script_path=JOB_SCRIPT,
                )
                arm_count += 1

    rel = RECIPE_DIR.relative_to(REPO_ROOT)
    print(f"Generated {anchor_count} anchor recipes and {arm_count} GPU arm recipes under {rel}")


if __name__ == "__main__":
    main()

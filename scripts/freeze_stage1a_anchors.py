#!/usr/bin/env python3
"""Verify, freeze, and SHA-256 hash all 21 Stage-1A vertex-specialist anchor checkpoints.

Enforces the Gate-1/Gate-1B anchor_freeze_before_evaluation_rule from
configs/protocols/stage1_simplex_face_exposure_v1.json:
  - All 21 final fixed-budget (4M-interaction) anchor checkpoints (9 Minecart + 12 Lunar)
    must be present on disk, match their SHA-256 checksums, and carry final_test_steps == 0.
  - Emits research/evidence/stage1a_vertex_anchors_frozen_v1.json with full provenance so
    Stage-1A arm evaluation scripts can refuse to run unless this manifest is verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "protocols" / "stage1_simplex_face_exposure_v1.json"
DEFAULT_OUT = REPO_ROOT / "research" / "evidence" / "stage1a_vertex_anchors_frozen_v1.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(PROTOCOL_PATH.relative_to(REPO_ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(REPO_ROOT)))
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Inspect current progress without writing the frozen manifest (exits 0 with status summary).",
    )
    args = parser.parse_args()

    protocol_path = REPO_ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    benchmarks = {b["env_id"]: b for b in protocol["benchmarks"]}

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    errors: list[str] = []
    per_env_summary: dict[str, Any] = {}

    for env_id in ("minecart-v0", "mo-lunar-lander-v3"):
        bench = benchmarks[env_id]
        expected_steps = int(bench["training_interactions"])
        n_vertices = int(bench["anchor_spec"]["vertices"])
        anchor_seeds = [int(s) for s in bench["anchor_spec"]["anchor_seeds"]]
        anchor_dir = REPO_ROOT / "artifacts" / "stage1a-gpu" / "anchors" / env_id

        env_vertices: dict[str, Any] = {}
        for v in range(n_vertices):
            v_argmax_returns: list[list[float]] = []
            v_sampled_returns: list[list[float]] = []
            v_records: list[dict[str, Any]] = []
            for seed in anchor_seeds:
                tag = f"{env_id}_anchor_vertex{v}_seed{seed}"
                ckpt_path = anchor_dir / f"{tag}.pt"
                receipt_path = anchor_dir / f"{tag}_receipt.json"
                if not ckpt_path.exists() or not receipt_path.exists():
                    missing.append(tag)
                    continue
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                actual_sha = _file_sha256(ckpt_path)
                if actual_sha != receipt.get("checkpoint_sha256"):
                    errors.append(f"{tag}: checkpoint SHA-256 mismatch")
                if int(receipt.get("training_interactions", -1)) != expected_steps:
                    errors.append(
                        f"{tag}: training_interactions={receipt.get('training_interactions')} != {expected_steps}"
                    )
                if int(receipt.get("final_test_steps", -1)) != 0:
                    errors.append(f"{tag}: final_test_steps != 0")

                anchor_eval = receipt.get("anchor_dev_eval") or {}
                modes = anchor_eval.get("modes") or {}
                if "argmax" in modes:
                    v_argmax_returns.append(modes["argmax"]["mean_scaled_discounted_return"])
                if "sampled" in modes:
                    v_sampled_returns.append(modes["sampled"]["mean_scaled_discounted_return"])

                rec = {
                    "tag": tag,
                    "env_id": env_id,
                    "vertex_index": v,
                    "seed": seed,
                    "hei_run_id": receipt.get("hei_run_id"),
                    "training_interactions": int(receipt["training_interactions"]),
                    "actor_updates": int(receipt["actor_updates"]),
                    "critic_updates": int(receipt["critic_updates"]),
                    "checkpoint_relpath": str(ckpt_path.relative_to(REPO_ROOT)),
                    "checkpoint_sha256": actual_sha,
                    "final_model_sha256": receipt.get("final_model_sha256"),
                    "gpu_hours": receipt.get("gpu_hours"),
                    "runtime": receipt.get("runtime"),
                    "final_test_steps": int(receipt["final_test_steps"]),
                    "anchor_dev_eval": anchor_eval,
                }
                records.append(rec)
                v_records.append(rec)

            env_vertices[f"vertex_{v}"] = {
                "completed": len(v_records),
                "expected": len(anchor_seeds),
                "argmax_scaled_discounted_returns": v_argmax_returns,
                "sampled_scaled_discounted_returns": v_sampled_returns,
                "argmax_mean_scaled_return": (np.mean(v_argmax_returns, axis=0).tolist() if v_argmax_returns else None),
                "argmax_best_vertex_utility": (
                    float(max(r[v] for r in v_argmax_returns)) if v_argmax_returns else None
                ),
            }
        per_env_summary[env_id] = {
            "reward_dim": n_vertices,
            "expected_anchors": n_vertices * len(anchor_seeds),
            "completed_anchors": sum(item["completed"] for item in env_vertices.values()),
            "vertices": env_vertices,
        }

    total_expected = 21
    print(
        f"Stage-1A vertex anchors: {len(records)}/{total_expected} complete, "
        f"{len(missing)} missing, {len(errors)} errors"
    )
    if args.allow_partial:
        if missing:
            print("Missing:", ", ".join(missing))
        if errors:
            print("Errors:", "; ".join(errors))
        return

    if missing or errors or len(records) != total_expected:
        raise SystemExit(
            f"Refusing to freeze Stage-1A anchors: {len(records)}/{total_expected} complete, "
            f"missing={missing[:5]}, errors={errors[:5]}"
        )

    manifest_body = {
        "schema_version": 1,
        "artifact_id": "stage1a_vertex_anchors_frozen_v1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_revision": protocol["protocol_revision"],
        "protocol_sha256": _file_sha256(protocol_path),
        "total_anchors": len(records),
        "final_test_steps": 0,
        "per_environment": per_env_summary,
        "anchors": records,
    }
    canonical = json.dumps(manifest_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest_body["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest_body, indent=2) + "\n", encoding="utf-8")
    print(
        f"Froze 21 Stage-1A vertex anchors to {out_path} (manifest_sha256={manifest_body['manifest_sha256'][:16]}...)"
    )


if __name__ == "__main__":
    main()

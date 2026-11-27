#!/usr/bin/env python3
"""Evaluate frozen Stage-1A GPU arm checkpoints on development splits (final_test_steps = 0).

Hard protocol gates enforced before running:
  1. Verifies research/evidence/stage1a_vertex_anchors_frozen_v1.json (cryptographic manifest_sha256,
     total_anchors == 21, final_test_steps == 0). Refuses to run if anchors are not frozen first.
  2. Evaluates ONLY on each benchmark's development_seed split (strictly never final_test_seed),
     recording final_test_steps = 0.
  3. Reports unclamped mean signed gap (primary) + cvar_10_signed_gap + fraction_clipped,
     reference-shifted interior utility retention (95% paired bootstrap CI lower bound >= -0.015),
     separate task success rate (ore_delivery_rate / native_plus100_helipad_landing_rate),
     Spearman controllability, sign-guarded APM (magnitude + reference-shifted), Effective Coverage C,
     Archive Blindness gap, hypervolume, and exact paired sign-flip permutation p-values + Cohen's d_z.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.envs.wrapper import make_mo_env  # noqa: E402
from src.evaluation.metrics import compute_expected_utility, compute_hypervolume, pareto_prune  # noqa: E402
from src.evaluation.simplex_metrics import (  # noqa: E402
    compute_checkpoint_metrics_bundle,
    compute_shifted_interior_retention,
    sample_edge_cap_preferences,
    sample_uniform_simplex,
    sample_vertex_cap_preferences,
)
from src.evaluation.stats import bootstrap_ci, paired_permutation_test  # noqa: E402
from src.models.onpolicy import ActorCritic  # noqa: E402
from src.utils.provenance import collect_provenance, write_json  # noqa: E402

PROTOCOL_PATH = REPO_ROOT / "configs" / "protocols" / "stage1_simplex_face_exposure_v1.json"
ANCHOR_MANIFEST = REPO_ROOT / "research" / "evidence" / "stage1a_vertex_anchors_frozen_v1.json"
DEFAULT_OUT = REPO_ROOT / "research" / "evidence" / "stage1a_gpu_dev_v1.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_anchor_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Frozen anchor manifest missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    recorded = data.get("manifest_sha256")
    body = {k: v for k, v in data.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if (
        recorded != hashlib.sha256(canonical).hexdigest()
        or int(data.get("total_anchors", 0)) != 21
        or int(data.get("final_test_steps", -1)) != 0
    ):
        raise RuntimeError(f"Frozen anchor manifest integrity check failed: {path}")
    return data


def _iqm(values: np.ndarray) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    if len(arr) < 4:
        return float(np.mean(arr))
    q1 = math.floor(0.25 * len(arr))
    q3 = math.ceil(0.75 * len(arr))
    return float(np.mean(arr[q1:q3]))


def _paired_summary(a: list[float], b: list[float], seed: int) -> dict[str, Any]:
    """Paired contrast of arm a minus arm b across matched training seeds."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    diffs = arr_a - arr_b
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    dz = mean_diff / std_diff if std_diff > 1e-12 else 0.0
    _, ci_lo, ci_hi = bootstrap_ci(diffs, n_boot=10_000, seed=seed)
    perm = paired_permutation_test(arr_a, arr_b, seed=seed)
    return {
        "n_seeds": len(diffs),
        "arm_a_mean": float(np.mean(arr_a)),
        "arm_b_mean": float(np.mean(arr_b)),
        "arm_a_iqm": _iqm(arr_a),
        "arm_b_iqm": _iqm(arr_b),
        "mean_diff_a_minus_b": mean_diff,
        "ci95_diff": [float(ci_lo), float(ci_hi)],
        "cohens_dz": float(dz),
        "paired_permutation_p": float(perm["p_value"]),
        "paired_permutation_mode": str(perm["mode"]),
    }


def build_dev_splits(bench: dict[str, Any]) -> dict[str, np.ndarray]:
    """Build deterministic development preference splits (never touches final_test_seed)."""
    d = int(bench["reward_dim"])
    es = bench["evaluation_splits"]
    rng = np.random.default_rng(int(es["development_seed"]))
    eps = float(es["epsilon"])
    counts = es["counts"]
    vertices = np.eye(d, dtype=np.float32)
    vertex_cap = sample_vertex_cap_preferences(d, per_vertex=int(counts["vertex_cap"]) // d, epsilon=eps, rng=rng)
    n_edges = d * (d - 1) // 2
    edge_cap = sample_edge_cap_preferences(d, per_edge=int(counts["edge_cap"]) // n_edges, epsilon=eps, rng=rng)
    non_vertex_face_cap = np.vstack([vertex_cap, edge_cap])
    interior_uniform = sample_uniform_simplex(d, count=int(counts["interior_uniform"]), rng=rng)
    return {
        "vertices": vertices,
        "vertex_cap": vertex_cap,
        "edge_cap": edge_cap,
        "non_vertex_face_cap": non_vertex_face_cap,
        "interior_uniform": interior_uniform,
    }


def build_reference_front(bench: dict[str, Any], anchor_manifest: dict[str, Any], mode: str) -> np.ndarray:
    """Build the frozen approximate reference front from frozen anchors (+ frozen Minecart front)."""
    env_id = str(bench["env_id"])
    scale = np.asarray(bench["reward_scale"], dtype=np.float64)
    key = "argmax_scaled_discounted_returns" if mode == "argmax" else "sampled_scaled_discounted_returns"
    points: list[np.ndarray] = []

    env_anchors = anchor_manifest["per_environment"][env_id]["vertices"]
    for v_data in env_anchors.values():
        for pt in v_data[key]:
            points.append(np.asarray(pt, dtype=np.float64))

    if env_id == "minecart-v0":
        ref_path = REPO_ROOT / str(bench["reference_front_artifact"])
        if _file_sha256(ref_path) != bench["reference_front_sha256"]:
            raise RuntimeError(f"Minecart reference front SHA-256 mismatch: {ref_path}")
        raw_front = np.asarray(json.loads(ref_path.read_text(encoding="utf-8"))["points"], dtype=np.float64)
        for pt in raw_front * scale:
            points.append(pt)

    return pareto_prune(np.asarray(points, dtype=np.float64))


@torch.no_grad()
def evaluate_checkpoint_on_splits(
    ckpt_path: Path,
    bench: dict[str, Any],
    splits: dict[str, np.ndarray],
    device: str,
) -> dict[str, Any]:
    """Evaluate one frozen GPU checkpoint on development splits in argmax and sampled modes."""
    env_id = str(bench["env_id"])
    d = int(bench["reward_dim"])
    state_dim = 7 if env_id == "minecart-v0" else 8
    action_dim = 6 if env_id == "minecart-v0" else 4
    width = int(bench["width"])
    gamma = float(bench["gamma"])
    scale = np.asarray(bench["reward_scale"], dtype=np.float64)
    episodes_per_pref = int(bench["evaluation_splits"]["episodes_per_preference"])
    dev_seed = int(bench["evaluation_splits"]["development_seed"])

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)  # nosec B614
    model = ActorCritic(state_dim=state_dim, action_dim=action_dim, reward_dim=d, width=width).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    env = make_mo_env(env_id)
    out: dict[str, Any] = {"dev_eval_steps": 0, "modes": {}}
    try:
        for mode in ("argmax", "sampled"):
            mode_res: dict[str, Any] = {}
            for split_name, prefs in splits.items():
                scaled_means = np.zeros((len(prefs), d), dtype=np.float64)
                raw_means = np.zeros((len(prefs), d), dtype=np.float64)
                successes: list[bool] = []
                for p_idx, pref in enumerate(prefs):
                    w_t = torch.tensor(pref, dtype=torch.float32, device=device).unsqueeze(0)
                    ep_scaled = np.zeros((episodes_per_pref, d), dtype=np.float64)
                    ep_raw = np.zeros((episodes_per_pref, d), dtype=np.float64)
                    for rep in range(episodes_per_pref):
                        ep_seed = dev_seed + p_idx * 17 + rep
                        torch.manual_seed(ep_seed)
                        np.random.seed(ep_seed)
                        obs, _ = env.reset(seed=ep_seed)
                        raw = np.zeros(d, dtype=np.float64)
                        disc = np.zeros(d, dtype=np.float64)
                        discount = 1.0
                        terminated = False
                        for _ in range(1000):
                            s_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                            dist = model.distribution(s_t, w_t)
                            act = int(dist.logits.argmax().item() if mode == "argmax" else dist.sample().item())
                            obs, r, tm, tr, _ = env.step(act)
                            rv = np.asarray(r, dtype=np.float64)
                            raw += rv
                            disc += discount * rv
                            discount *= gamma
                            out["dev_eval_steps"] += 1
                            if tm or tr:
                                terminated = bool(tm)
                                break
                        ep_raw[rep] = raw
                        ep_scaled[rep] = disc * scale
                        if env_id == "minecart-v0":
                            successes.append(bool(np.any(raw[:2] > 0.0)))
                        else:
                            unw: Any = env.unwrapped
                            inside_pad = bool(
                                hasattr(unw, "helipad_x1")
                                and hasattr(unw, "helipad_x2")
                                and hasattr(unw, "lander")
                                and unw.lander is not None
                                and unw.helipad_x1 <= unw.lander.position.x <= unw.helipad_x2
                            )
                            successes.append(bool(terminated and raw[0] >= 99.9 and inside_pad))
                    raw_means[p_idx] = ep_raw.mean(axis=0)
                    scaled_means[p_idx] = ep_scaled.mean(axis=0)
                mode_res[split_name] = {
                    "scaled_returns": scaled_means,
                    "raw_returns": raw_means,
                    "task_success_rate": float(np.mean(successes)) if successes else 0.0,
                }
            out["modes"][mode] = mode_res
    finally:
        env.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(PROTOCOL_PATH.relative_to(REPO_ROOT)))
    parser.add_argument("--anchors", default=str(ANCHOR_MANIFEST.relative_to(REPO_ROOT)))
    parser.add_argument("--env", choices=["all", "minecart-v0", "mo-lunar-lander-v3"], default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(REPO_ROOT)))
    args = parser.parse_args()

    protocol_path = REPO_ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    anchor_manifest = _verify_anchor_manifest(REPO_ROOT / args.anchors)
    benchmarks = {b["env_id"]: b for b in protocol["benchmarks"]}
    env_ids = ["minecart-v0", "mo-lunar-lander-v3"] if args.env == "all" else [args.env]

    t0 = time.monotonic()
    total_dev_eval_steps = 0
    env_results: dict[str, Any] = {}

    for env_id in env_ids:
        bench = benchmarks[env_id]
        splits = build_dev_splits(bench)
        ref_point = np.asarray(bench["reference_point"], dtype=np.float64)
        arm_dir = REPO_ROOT / "artifacts" / "stage1a-gpu" / "arms" / env_id
        ref_fronts = {m: build_reference_front(bench, anchor_manifest, m) for m in ("argmax", "sampled")}

        per_seed_records: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in bench["arms_included"]}
        per_seed_interior: dict[str, dict[int, dict[str, np.ndarray]]] = {arm: {} for arm in bench["arms_included"]}

        for seed in bench["paired_seeds"]:
            for arm in bench["arms_included"]:
                tag = f"{env_id}_arm_{arm}_seed{seed}"
                ckpt = arm_dir / f"{tag}.pt"
                receipt_path = arm_dir / f"{tag}_receipt.json"
                if not ckpt.exists() or not receipt_path.exists():
                    raise RuntimeError(f"Missing frozen Stage-1A arm checkpoint or receipt: {tag}")
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if _file_sha256(ckpt) != receipt.get("checkpoint_sha256"):
                    raise RuntimeError(f"Checkpoint SHA-256 mismatch for {tag}")

                raw_eval = evaluate_checkpoint_on_splits(ckpt, bench, splits, args.device)
                total_dev_eval_steps += int(raw_eval["dev_eval_steps"])

                mode_metrics: dict[str, Any] = {}
                per_seed_interior[arm][int(seed)] = {}
                for mode in ("argmax", "sampled"):
                    nv_ret = raw_eval["modes"][mode]["non_vertex_face_cap"]["scaled_returns"]
                    in_ret = raw_eval["modes"][mode]["interior_uniform"]["scaled_returns"]
                    vx_ret = raw_eval["modes"][mode]["vertices"]["scaled_returns"]
                    per_seed_interior[arm][int(seed)][mode] = in_ret

                    nv_bundle = compute_checkpoint_metrics_bundle(
                        nv_ret, splits["non_vertex_face_cap"], ref_fronts[mode], reference_point=ref_point, tau=0.1
                    )
                    vx_bundle = compute_checkpoint_metrics_bundle(
                        vx_ret, splits["vertices"], ref_fronts[mode], reference_point=ref_point, tau=0.1
                    )
                    in_bundle = compute_checkpoint_metrics_bundle(
                        in_ret, splits["interior_uniform"], ref_fronts[mode], reference_point=ref_point, tau=0.1
                    )
                    combined_ret = np.vstack([in_ret, nv_ret, vx_ret])
                    mode_metrics[mode] = {
                        "non_vertex_face_cap": nv_bundle,
                        "vertices": vx_bundle,
                        "interior_uniform": in_bundle,
                        "interior_expected_utility": float(
                            compute_expected_utility(in_ret, splits["interior_uniform"])
                        ),
                        "hypervolume_combined": float(compute_hypervolume(combined_ret, ref_point)),
                        "task_success_rate_non_vertex_face_cap": raw_eval["modes"][mode]["non_vertex_face_cap"][
                            "task_success_rate"
                        ],
                        "task_success_rate_interior": raw_eval["modes"][mode]["interior_uniform"]["task_success_rate"],
                    }
                per_seed_records[arm][int(seed)] = {
                    "tag": tag,
                    "checkpoint_sha256": receipt["checkpoint_sha256"],
                    "hei_run_id": receipt.get("hei_run_id"),
                    "hardware": receipt.get("runtime", {}).get("hardware"),
                    "gpu_hours": receipt.get("gpu_hours"),
                    "modes": mode_metrics,
                }

        # Paired contrasts (primary: fixed_face_balanced vs uniform in argmax mode).
        seeds = [int(s) for s in bench["paired_seeds"]]
        contrasts: dict[str, Any] = {}
        pairs = [("fixed_face_balanced", "uniform")]
        if "dirichlet_sparse" in bench["arms_included"]:
            pairs.extend([("fixed_face_balanced", "dirichlet_sparse"), ("dirichlet_sparse", "uniform")])

        for arm_a, arm_b in pairs:
            cname = f"{arm_a}_vs_{arm_b}"
            c_modes: dict[str, Any] = {}
            for mode in ("argmax", "sampled"):
                a_gap = [
                    per_seed_records[arm_a][s]["modes"][mode]["non_vertex_face_cap"]["deployed_regret"][
                        "mean_signed_gap"
                    ]
                    for s in seeds
                ]
                b_gap = [
                    per_seed_records[arm_b][s]["modes"][mode]["non_vertex_face_cap"]["deployed_regret"][
                        "mean_signed_gap"
                    ]
                    for s in seeds
                ]
                a_util = [
                    per_seed_records[arm_a][s]["modes"][mode]["non_vertex_face_cap"]["deployed_regret"][
                        "mean_deployed_utility"
                    ]
                    for s in seeds
                ]
                b_util = [
                    per_seed_records[arm_b][s]["modes"][mode]["non_vertex_face_cap"]["deployed_regret"][
                        "mean_deployed_utility"
                    ]
                    for s in seeds
                ]
                a_clip = [
                    per_seed_records[arm_a][s]["modes"][mode]["non_vertex_face_cap"]["deployed_regret"][
                        "fraction_clipped"
                    ]
                    for s in seeds
                ]
                b_clip = [
                    per_seed_records[arm_b][s]["modes"][mode]["non_vertex_face_cap"]["deployed_regret"][
                        "fraction_clipped"
                    ]
                    for s in seeds
                ]
                ret_changes = [
                    compute_shifted_interior_retention(
                        per_seed_interior[arm_a][s][mode],
                        per_seed_interior[arm_b][s][mode],
                        splits["interior_uniform"],
                        ref_point,
                    )["shifted_relative_change"]
                    for s in seeds
                ]
                _, ret_lo, ret_hi = bootstrap_ci(np.asarray(ret_changes, dtype=np.float64), n_boot=10_000, seed=777)
                c_modes[mode] = {
                    "primary_non_vertex_mean_signed_gap": _paired_summary(a_gap, b_gap, seed=101),
                    "direct_paired_non_vertex_deployed_utility": _paired_summary(a_util, b_util, seed=202),
                    "mean_fraction_clipped_arm_a": float(np.mean(a_clip)),
                    "mean_fraction_clipped_arm_b": float(np.mean(b_clip)),
                    "shifted_interior_retention": {
                        "mean_relative_change": float(np.mean(ret_changes)),
                        "ci95_relative_change": [float(ret_lo), float(ret_hi)],
                        "passes_non_inferiority_ge_minus_0015": bool(ret_lo >= -0.015),
                    },
                }
            contrasts[cname] = c_modes

        env_results[env_id] = {
            "reward_dim": int(bench["reward_dim"]),
            "paired_seeds": seeds,
            "reference_front_sizes": {m: len(ref_fronts[m]) for m in ("argmax", "sampled")},
            "contrasts": contrasts,
            "per_arm_seed_records": per_seed_records,
        }

    out_payload = {
        "schema_version": 1,
        "artifact_id": "stage1a_gpu_dev_v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_revision": protocol["protocol_revision"],
        "protocol_sha256": _file_sha256(protocol_path),
        "anchor_manifest_sha256": anchor_manifest["manifest_sha256"],
        "final_test_steps": 0,
        "total_dev_eval_steps": total_dev_eval_steps,
        "eval_wall_seconds": round(time.monotonic() - t0, 2),
        "provenance": collect_provenance(
            {
                "protocol": args.protocol,
                "anchors": args.anchors,
                "env": args.env,
                "device": args.device,
                "final_test_steps": 0,
            }
        ),
        "benchmarks": env_results,
    }
    write_json(REPO_ROOT / args.out, out_payload)
    print(f"Wrote Stage-1A GPU development evidence to {args.out} (final_test_steps=0)")


if __name__ == "__main__":
    main()

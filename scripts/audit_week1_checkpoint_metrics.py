#!/usr/bin/env python3
"""Week-1 Step-2 audit: score 5 Minecart PPO checkpoints with Spearman, sign-guarded APM, and face-cap regret.

Zero training, zero final-test access. Verifies checkpoint SHA-256 against recorded evidence receipts,
evaluates all 5 checkpoints on identical non-test uniform and simplex face-cap preference splits,
explicitly documents the negative-utility APM sign edge case, and checks whether Spearman (2605.10585)
or sign-guarded APM (2609.26918) already rank checkpoints nearly identically (Spearman >= 0.9) to
deployed face-cap regret.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs.wrapper import make_mo_env
from src.evaluation.metrics import compute_expected_utility, compute_hypervolume
from src.evaluation.protocol import load_protocol, resolve_reference_point, resolve_reward_scale
from src.evaluation.simplex_metrics import (
    compute_abandoned_preference_mass,
    compute_checkpoint_metrics_bundle,
    compute_rank_correlations,
    sample_edge_cap_preferences,
    sample_vertex_cap_preferences,
)
from src.models.onpolicy import ActorCritic
from src.utils.provenance import collect_provenance, write_json

CHECKPOINT_SPECS: list[dict[str, str]] = [
    {
        "id": "minecart_S_20264111",
        "stage": "S",
        "label": "Specialist PPO [0.45, 0.45, 0.10] (seed 20264111)",
        "checkpoint": (
            "artifacts/minecart-onpolicy-specialist-v1/runs/minecart-v0_S_seed20264111/resume_step004000000.pt"
        ),
        "evidence": "research/evidence/minecart_onpolicy_specialist_v1.json",
    },
    {
        "id": "minecart_C_20264111",
        "stage": "C",
        "label": "Conditional PPO 32-pool (seed 20264111)",
        "checkpoint": (
            "artifacts/minecart-onpolicy-conditional-v1/runs/minecart-v0_C_seed20264111/resume_step004000000.pt"
        ),
        "evidence": "research/evidence/minecart_onpolicy_conditional_v1.json",
    },
    {
        "id": "minecart_G_20264311",
        "stage": "G",
        "label": "Confirmation PPO 32-pool (seed 20264311)",
        "checkpoint": (
            "artifacts/minecart-onpolicy-confirmation-v1/runs/minecart-v0_G_seed20264311/resume_step004000000.pt"
        ),
        "evidence": "research/evidence/minecart_onpolicy_confirmation_v1.json",
    },
    {
        "id": "minecart_G_20264312",
        "stage": "G",
        "label": "Confirmation PPO 32-pool (seed 20264312)",
        "checkpoint": (
            "artifacts/minecart-onpolicy-confirmation-v1/runs/minecart-v0_G_seed20264312/resume_step004000000.pt"
        ),
        "evidence": "research/evidence/minecart_onpolicy_confirmation_v1.json",
    },
    {
        "id": "minecart_G_20264313",
        "stage": "G",
        "label": "Confirmation PPO 32-pool (seed 20264313)",
        "checkpoint": (
            "artifacts/minecart-onpolicy-confirmation-v1/runs/minecart-v0_G_seed20264313/resume_step004000000.pt"
        ),
        "evidence": "research/evidence/minecart_onpolicy_confirmation_v1.json",
    },
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_checkpoint_sha(evidence_path: Path, checkpoint_rel: str) -> str:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    norm_target = checkpoint_rel.replace("\\", "/")
    if "files" in evidence:
        for key, value in evidence["files"].items():
            if key.replace("\\", "/") == norm_target:
                return str(value)
    for job in evidence.get("jobs", []):
        for key, value in job.get("files", {}).items():
            if key.replace("\\", "/") == norm_target:
                return str(value)
    raise ValueError(f"Could not find recorded SHA-256 for {checkpoint_rel} in {evidence_path}")


def build_minecart_eval_splits(
    protocol_path: str,
    face_cap_seed: int,
    epsilon: float,
    per_vertex: int,
    per_edge: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    protocol, validation_uniform = load_protocol(protocol_path, "validation")
    rng = np.random.default_rng(face_cap_seed)
    vertices = np.eye(3, dtype=np.float32)
    vertex_caps = sample_vertex_cap_preferences(3, per_vertex=per_vertex, epsilon=epsilon, rng=rng)
    edge_caps = sample_edge_cap_preferences(3, per_edge=per_edge, epsilon=epsilon, rng=rng)
    face_cap = np.vstack([vertices, vertex_caps, edge_caps])
    combined = np.vstack([validation_uniform, face_cap])
    splits = {
        "validation_uniform": validation_uniform,
        "vertices": vertices,
        "vertex_cap": vertex_caps,
        "edge_cap": edge_caps,
        "face_cap": face_cap,
        "combined": combined,
    }
    return protocol, splits


@torch.no_grad()
def evaluate_checkpoint_on_preferences(
    model: ActorCritic,
    preferences: np.ndarray,
    env_seeds: list[int],
    scale: np.ndarray,
    gamma: float,
    horizon: int,
    mode: str,
    device: str,
) -> dict[str, Any]:
    env = make_mo_env("minecart-v0")
    mean_scaled_returns = np.zeros((len(preferences), 3), dtype=np.float64)
    mean_raw_returns = np.zeros((len(preferences), 3), dtype=np.float64)
    delivery_rates = np.zeros(len(preferences), dtype=np.float64)
    total_steps = 0

    try:
        for pref_idx, pref in enumerate(preferences):
            w_tensor = torch.tensor(pref, dtype=torch.float32, device=device).unsqueeze(0)
            ep_scaled = np.zeros((len(env_seeds), 3), dtype=np.float64)
            ep_raw = np.zeros((len(env_seeds), 3), dtype=np.float64)
            delivered = 0
            for s_idx, env_seed in enumerate(env_seeds):
                case_seed = int(
                    np.random.SeedSequence([env_seed, pref_idx, 0 if mode == "argmax" else 1]).generate_state(1)[0]
                )
                np.random.seed(case_seed)
                torch.manual_seed(case_seed)
                obs, _ = env.reset(seed=env_seed)
                raw = np.zeros(3, dtype=np.float64)
                discounted = np.zeros(3, dtype=np.float64)
                discount = 1.0
                for _ in range(horizon):
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    dist = model.distribution(obs_t, w_tensor)
                    action = int(dist.logits.argmax().item() if mode == "argmax" else dist.sample().item())
                    obs, reward, terminated, truncated, _ = env.step(action)
                    r = np.asarray(reward, dtype=np.float64)
                    raw += r
                    discounted += discount * r
                    discount *= gamma
                    total_steps += 1
                    if terminated or truncated:
                        break
                ep_raw[s_idx] = raw
                ep_scaled[s_idx] = discounted * scale
                if np.any(raw[:2] > 0.0):
                    delivered += 1
            mean_raw_returns[pref_idx] = ep_raw.mean(axis=0)
            mean_scaled_returns[pref_idx] = ep_scaled.mean(axis=0)
            delivery_rates[pref_idx] = delivered / float(len(env_seeds))
    finally:
        env.close()

    return {
        "mean_scaled_returns": mean_scaled_returns,
        "mean_raw_returns": mean_raw_returns,
        "delivery_rates": delivery_rates,
        "total_steps": total_steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/protocols/minecart_confirmation_development_v1.json")
    parser.add_argument("--reference-front", default="research/calibration/minecart_v0_gamma099_reference_front.json")
    parser.add_argument("--face-cap-seed", type=int, default=20267101)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--per-vertex", type=int, default=6)
    parser.add_argument("--per-edge", type=int, default=6)
    parser.add_argument("--episodes-per-preference", type=int, default=3)
    parser.add_argument("--eval-seed-start", type=int, default=20267001)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", default="research/evidence/week1_minecart_checkpoint_metrics_v1.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    torch.set_num_threads(1)

    protocol, splits = build_minecart_eval_splits(
        args.protocol,
        face_cap_seed=args.face_cap_seed,
        epsilon=args.epsilon,
        per_vertex=args.per_vertex,
        per_edge=args.per_edge,
    )
    scale = resolve_reward_scale(protocol).astype(np.float64)
    ref_point_raw = resolve_reference_point(protocol)
    if ref_point_raw is None:
        raise ValueError("Minecart protocol must define a frozen reference_point")
    ref_point = ref_point_raw.astype(np.float64)
    gamma = float(protocol["return_definition"]["gamma"])
    env_seeds = [args.eval_seed_start + i for i in range(args.episodes_per_preference)]

    ref_artifact = json.loads(Path(args.reference_front).read_text(encoding="utf-8"))
    env_ref_front_scaled = np.asarray(ref_artifact["points"], dtype=np.float64) * scale

    # Explicitly witness the negative-utility APM sign edge case on the reference front itself.
    self_apm_witness = {
        split_name: {
            "raw_unguarded": compute_abandoned_preference_mass(
                env_ref_front_scaled,
                env_ref_front_scaled,
                splits[split_name],
                tau=args.tau,
                sign_guard="raw_unguarded",
            ),
            "magnitude_guarded": compute_abandoned_preference_mass(
                env_ref_front_scaled,
                env_ref_front_scaled,
                splits[split_name],
                tau=args.tau,
                sign_guard="magnitude",
            ),
            "reference_shifted": compute_abandoned_preference_mass(
                env_ref_front_scaled,
                env_ref_front_scaled,
                splits[split_name],
                tau=args.tau,
                sign_guard="reference_shifted",
                reference_point=ref_point,
            ),
        }
        for split_name in ("validation_uniform", "face_cap", "combined")
    }

    n_unif = len(splits["validation_uniform"])
    n_vert = len(splits["vertices"])
    n_vcap = len(splits["vertex_cap"])
    n_ecap = len(splits["edge_cap"])
    split_slices = {
        "validation_uniform": slice(0, n_unif),
        "vertices": slice(n_unif, n_unif + n_vert),
        "vertex_cap": slice(n_unif + n_vert, n_unif + n_vert + n_vcap),
        "edge_cap": slice(n_unif + n_vert + n_vcap, n_unif + n_vert + n_vcap + n_ecap),
        "face_cap": slice(n_unif, n_unif + n_vert + n_vcap + n_ecap),
        "combined": slice(0, n_unif + n_vert + n_vcap + n_ecap),
    }

    checkpoint_results: dict[str, Any] = {}
    all_returns_by_mode: dict[str, dict[str, np.ndarray]] = {"argmax": {}, "sampled": {}}
    total_eval_steps = 0

    for spec in CHECKPOINT_SPECS:
        ckpt_path = Path(spec["checkpoint"])
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing audited checkpoint: {ckpt_path}")
        actual_sha = file_sha256(ckpt_path)
        expected_sha = expected_checkpoint_sha(Path(spec["evidence"]), spec["checkpoint"])
        if actual_sha != expected_sha:
            raise ValueError(f"SHA-256 mismatch for {spec['id']}: {actual_sha} != {expected_sha}")

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # nosec B614: audited local checkpoint
        model = ActorCritic(state_dim=7, action_dim=6, reward_dim=3, width=256).to(args.device)
        model.load_state_dict(payload["learner"]["model"])
        model.eval()

        modes_out: dict[str, Any] = {}
        for mode in ("argmax", "sampled"):
            eval_out = evaluate_checkpoint_on_preferences(
                model=model,
                preferences=splits["combined"],
                env_seeds=env_seeds,
                scale=scale,
                gamma=gamma,
                horizon=1000,
                mode=mode,
                device=args.device,
            )
            total_eval_steps += int(eval_out["total_steps"])
            returns_all = eval_out["mean_scaled_returns"]
            all_returns_by_mode[mode][spec["id"]] = returns_all

            mode_splits: dict[str, Any] = {}
            for split_name in ("validation_uniform", "vertices", "vertex_cap", "edge_cap", "face_cap", "combined"):
                sl = split_slices[split_name]
                sub_returns = returns_all[sl]
                sub_prefs = splits[split_name]
                bundle = compute_checkpoint_metrics_bundle(
                    deployed_returns=sub_returns,
                    preferences=sub_prefs,
                    reference_front=env_ref_front_scaled,
                    reference_point=ref_point,
                    tau=args.tau,
                )
                bundle["expected_utility"] = compute_expected_utility(sub_returns, sub_prefs)
                bundle["hypervolume"] = compute_hypervolume(sub_returns, ref_point)
                bundle["mean_delivery_rate"] = float(np.mean(eval_out["delivery_rates"][sl]))
                mode_splits[split_name] = bundle

            modes_out[mode] = {
                "evaluation_steps": int(eval_out["total_steps"]),
                "splits": mode_splits,
                "mean_scaled_returns_combined": returns_all.tolist(),
                "delivery_rates_combined": eval_out["delivery_rates"].tolist(),
            }

        checkpoint_results[spec["id"]] = {
            "id": spec["id"],
            "stage": spec["stage"],
            "label": spec["label"],
            "checkpoint": spec["checkpoint"],
            "checkpoint_sha256": actual_sha,
            "verified_against_evidence": spec["evidence"],
            "modes": modes_out,
        }
        am = modes_out["argmax"]["splits"]
        sm = modes_out["sampled"]["splits"]
        print(
            f"[{spec['id']}] argmax: face_cap_regret={am['face_cap']['deployed_regret']['mean_regret']:.4f} "
            f"spearman_unif={am['validation_uniform']['controllability']['mean_spearman']:.4f} "
            f"apm_mag_comb={am['combined']['apm_magnitude']['apm']:.4f} | "
            f"sampled: face_cap_regret={sm['face_cap']['deployed_regret']['mean_regret']:.4f} "
            f"spearman_unif={sm['validation_uniform']['controllability']['mean_spearman']:.4f} "
            f"apm_mag_comb={sm['combined']['apm_magnitude']['apm']:.4f}",
            flush=True,
        )

    # Also compute APM relative to checkpoint C (minecart_C_20264111) as baseline front F0, matching Bonin et al.
    ids = [spec["id"] for spec in CHECKPOINT_SPECS]
    ranking_analysis: dict[str, Any] = {}
    for mode in ("argmax", "sampled"):
        baseline_c_front = all_returns_by_mode[mode]["minecart_C_20264111"]
        apm_vs_c_mag: list[float] = []
        apm_vs_c_shifted: list[float] = []
        for ckpt_id in ids:
            front = all_returns_by_mode[mode][ckpt_id]
            res_mag = compute_abandoned_preference_mass(
                front, baseline_c_front, splits["combined"], tau=args.tau, sign_guard="magnitude"
            )
            res_shifted = compute_abandoned_preference_mass(
                front,
                baseline_c_front,
                splits["combined"],
                tau=args.tau,
                sign_guard="reference_shifted",
                reference_point=ref_point,
            )
            apm_vs_c_mag.append(res_mag["apm"])
            apm_vs_c_shifted.append(res_shifted["apm"])
            checkpoint_results[ckpt_id]["modes"][mode]["apm_vs_conditional_C_combined"] = {
                "magnitude": res_mag,
                "reference_shifted": res_shifted,
            }

        higher_is_better = {
            "neg_deployed_face_cap_regret": [
                -checkpoint_results[i]["modes"][mode]["splits"]["face_cap"]["deployed_regret"]["mean_regret"]
                for i in ids
            ],
            "neg_deployed_vertex_regret": [
                -checkpoint_results[i]["modes"][mode]["splits"]["vertices"]["deployed_regret"]["mean_regret"]
                for i in ids
            ],
            "neg_deployed_combined_regret": [
                -checkpoint_results[i]["modes"][mode]["splits"]["combined"]["deployed_regret"]["mean_regret"]
                for i in ids
            ],
            "mean_spearman_uniform": [
                checkpoint_results[i]["modes"][mode]["splits"]["validation_uniform"]["controllability"]["mean_spearman"]
                for i in ids
            ],
            "mean_spearman_combined": [
                checkpoint_results[i]["modes"][mode]["splits"]["combined"]["controllability"]["mean_spearman"]
                for i in ids
            ],
            "neg_apm_mag_uniform": [
                -checkpoint_results[i]["modes"][mode]["splits"]["validation_uniform"]["apm_magnitude"]["apm"]
                for i in ids
            ],
            "neg_apm_mag_combined": [
                -checkpoint_results[i]["modes"][mode]["splits"]["combined"]["apm_magnitude"]["apm"] for i in ids
            ],
            "neg_apm_shifted_combined": [
                -checkpoint_results[i]["modes"][mode]["splits"]["combined"]["apm_reference_shifted"]["apm"] for i in ids
            ],
            "neg_apm_vs_C_mag_combined": [-v for v in apm_vs_c_mag],
            "effective_coverage_combined": [
                checkpoint_results[i]["modes"][mode]["splits"]["combined"]["effective_coverage"] for i in ids
            ],
            "expected_utility_uniform": [
                checkpoint_results[i]["modes"][mode]["splits"]["validation_uniform"]["expected_utility"] for i in ids
            ],
            "hypervolume_combined": [
                checkpoint_results[i]["modes"][mode]["splits"]["combined"]["hypervolume"] for i in ids
            ],
        }
        correlations = compute_rank_correlations(higher_is_better)
        sp = correlations["spearman"]["neg_deployed_face_cap_regret"]
        gate_checks = {
            "face_cap_vs_spearman_uniform_rho": sp["mean_spearman_uniform"],
            "face_cap_vs_spearman_combined_rho": sp["mean_spearman_combined"],
            "face_cap_vs_apm_mag_uniform_rho": sp["neg_apm_mag_uniform"],
            "face_cap_vs_apm_mag_combined_rho": sp["neg_apm_mag_combined"],
            "face_cap_vs_apm_shifted_combined_rho": sp["neg_apm_shifted_combined"],
            "face_cap_vs_apm_vs_C_combined_rho": sp["neg_apm_vs_C_mag_combined"],
            "face_cap_vs_eu_uniform_rho": sp["expected_utility_uniform"],
            "face_cap_vs_hv_combined_rho": sp["hypervolume_combined"],
            "spearman_below_0_9_threshold": (
                sp["mean_spearman_uniform"] < 0.9
                and sp["mean_spearman_combined"] < 0.9
                and sp["neg_apm_mag_uniform"] < 0.9
                and sp["neg_apm_mag_combined"] < 0.9
            ),
        }
        ranking_analysis[mode] = {
            "checkpoint_order": ids,
            "higher_is_better_scores": higher_is_better,
            "correlations": correlations,
            "gate_checks": gate_checks,
        }

    report = {
        "schema_version": 1,
        "study_id": "week1-minecart-checkpoint-metrics-v1",
        "scope": (
            "Week-1 Step-2 metric differentiation audit on 5 audited Minecart PPO checkpoints; "
            "zero training, zero final-test access"
        ),
        "final_test_steps": 0,
        "training_interactions": 0,
        "total_evaluation_steps": total_eval_steps,
        "wall_seconds": round(time.monotonic() - started, 3),
        "config": {
            "protocol": args.protocol,
            "reference_front": args.reference_front,
            "gamma": gamma,
            "reward_scale": scale.tolist(),
            "reference_point": ref_point.tolist(),
            "face_cap_seed": args.face_cap_seed,
            "epsilon": args.epsilon,
            "per_vertex": args.per_vertex,
            "per_edge": args.per_edge,
            "env_seeds": env_seeds,
            "apm_tau": args.tau,
            "split_counts": {k: len(v) for k, v in splits.items()},
        },
        "apm_negative_utility_sign_guard_witness": self_apm_witness,
        "checkpoints": checkpoint_results,
        "ranking_analysis": ranking_analysis,
        "provenance": collect_provenance(vars(args)),
    }
    output_path = Path(args.output)
    write_json(output_path, report)
    print(f"Wrote {output_path} ({total_eval_steps} eval steps in {report['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()

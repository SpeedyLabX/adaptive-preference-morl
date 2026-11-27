#!/usr/bin/env python3
"""Week-1 Step-2 matched-budget paired-seed pilot on fruit-tree-v0 (d=6, exact discounted oracle).

Contrasts Uniform (Dirichlet(1_6)) vs. static Fixed Face-Balanced exposure using an identical
on-policy categorical PPO learner with PopArt vector critic, faithful raw-utility-preserving
scalarized advantage, and NO hindsight relabeling.

All runs share matched budgets, paired seeds, identical initial weights, and identical hardware.
Evaluates against the exact discounted oracle on fresh non-test vertex, vertex-cap, edge-cap,
and interior-uniform splits, plus Spearman (2605.10585) and sign-guarded APM (2609.26918).
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs.fruit_tree_oracle import encode_fruit_tree_node, exact_oracle_utility, extract_fruit_tree_oracle
from src.envs.wrapper import make_mo_env
from src.evaluation.metrics import compute_expected_utility
from src.evaluation.simplex_metrics import (
    FaceBalancedPreferenceSampler,
    compute_abandoned_preference_mass,
    compute_checkpoint_metrics_bundle,
    sample_edge_cap_preferences,
    sample_uniform_simplex,
    sample_vertex_cap_preferences,
)
from src.evaluation.stats import bootstrap_ci, holm_adjust, paired_permutation_test
from src.models.face_exposure_ppo import SimplexPPO
from src.models.onpolicy import ActorCritic
from src.utils.provenance import collect_provenance, write_json


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def build_fruit_tree_eval_splits(
    seed: int,
    epsilon: float = 0.10,
    per_vertex: int = 10,
    per_edge: int = 4,
    interior_count: int = 100,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    vertices = np.eye(6, dtype=np.float32)
    vertex_cap = sample_vertex_cap_preferences(6, per_vertex=per_vertex, epsilon=epsilon, rng=rng)
    edge_cap = sample_edge_cap_preferences(6, per_edge=per_edge, epsilon=epsilon, rng=rng)
    interior_uniform = sample_uniform_simplex(6, count=interior_count, rng=rng)
    face_cap = np.vstack([vertices, vertex_cap, edge_cap])
    combined = np.vstack([interior_uniform, face_cap])
    return {
        "vertices": vertices,
        "vertex_cap": vertex_cap,
        "edge_cap": edge_cap,
        "face_cap": face_cap,
        "interior_uniform": interior_uniform,
        "combined": combined,
    }


@torch.no_grad()
def evaluate_policy_on_fruit_tree(
    model: ActorCritic,
    splits: dict[str, np.ndarray],
    oracle_leaves: np.ndarray,
    gamma: float,
    tree_depth: int,
    device: str,
    sampled_rollouts: int = 5,
    eval_seed: int = 20268901,
) -> dict[str, Any]:
    """Evaluate a locked policy in argmax (exact deterministic) and sampled modes across all splits."""
    env = make_mo_env("fruit-tree-v0")
    ref_point = np.zeros(6, dtype=np.float64)
    n_int = len(splits["interior_uniform"])
    n_vert = len(splits["vertices"])
    n_vcap = len(splits["vertex_cap"])
    n_ecap = len(splits["edge_cap"])
    split_slices = {
        "interior_uniform": slice(0, n_int),
        "vertices": slice(n_int, n_int + n_vert),
        "vertex_cap": slice(n_int + n_vert, n_int + n_vert + n_vcap),
        "edge_cap": slice(n_int + n_vert + n_vcap, n_int + n_vert + n_vcap + n_ecap),
        "face_cap": slice(n_int, n_int + n_vert + n_vcap + n_ecap),
        "combined": slice(0, n_int + n_vert + n_vcap + n_ecap),
    }

    combined_prefs = splits["combined"]
    _, opt_leaves_combined = exact_oracle_utility(combined_prefs, oracle_leaves)
    modes_out: dict[str, Any] = {}
    total_eval_steps = 0

    try:
        for mode in ("argmax", "sampled"):
            n_reps = 1 if mode == "argmax" else sampled_rollouts
            mean_returns = np.zeros((len(combined_prefs), 6), dtype=np.float64)
            opt_hits = np.zeros(len(combined_prefs), dtype=np.float64)
            for idx, pref in enumerate(combined_prefs):
                w_t = torch.tensor(pref, dtype=torch.float32, device=device).unsqueeze(0)
                ep_returns = np.zeros((n_reps, 6), dtype=np.float64)
                hits = 0
                for rep in range(n_reps):
                    if mode == "sampled":
                        torch.manual_seed(eval_seed + idx * 100 + rep)
                    obs, _ = env.reset(seed=0)
                    discounted = np.zeros(6, dtype=np.float64)
                    discount = 1.0
                    for _ in range(tree_depth):
                        s_t = torch.tensor(
                            encode_fruit_tree_node(obs, tree_depth), dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        dist = model.distribution(s_t, w_t)
                        action = int(dist.logits.argmax().item() if mode == "argmax" else dist.sample().item())
                        obs, reward, terminated, _, _ = env.step(action)
                        discounted += discount * np.asarray(reward, dtype=np.float64)
                        discount *= gamma
                        total_eval_steps += 1
                        if terminated:
                            break
                    ep_returns[rep] = discounted
                    if int(obs[1]) == int(opt_leaves_combined[idx]):
                        hits += 1
                mean_returns[idx] = ep_returns.mean(axis=0)
                opt_hits[idx] = hits / float(n_reps)

            split_metrics: dict[str, Any] = {}
            for split_name, sl in split_slices.items():
                sub_ret = mean_returns[sl]
                sub_pref = splits[split_name]
                bundle = compute_checkpoint_metrics_bundle(
                    deployed_returns=sub_ret,
                    preferences=sub_pref,
                    reference_front=oracle_leaves,
                    reference_point=ref_point,
                    tau=0.10,
                )
                bundle["expected_utility"] = compute_expected_utility(sub_ret, sub_pref)
                bundle["exact_optimal_leaf_rate"] = float(np.mean(opt_hits[sl]))
                split_metrics[split_name] = bundle

            modes_out[mode] = {
                "splits": split_metrics,
                "mean_returns_combined": mean_returns.tolist(),
            }
    finally:
        env.close()

    return {"modes": modes_out, "evaluation_steps": total_eval_steps}


def train_single_run(
    arm: str,
    seed: int,
    initial_state_dict: dict[str, torch.Tensor],
    plan: dict[str, Any],
    splits: dict[str, np.ndarray],
    oracle_leaves: np.ndarray,
    device: str,
    artifact_dir: Path | None,
) -> dict[str, Any]:
    """Train one matched-budget on-policy SimplexPPO run (no hindsight relabeling) and evaluate."""
    started = time.monotonic()
    seed_all(seed)

    tree_depth = int(plan["tree_depth"])
    state_dim = (1 << (tree_depth + 1)) - 1
    streams = int(plan["streams"])
    fragment_steps = int(plan["fragment_steps"])
    total_steps = int(plan["training_interactions"])
    gamma = float(plan["gamma"])

    model = ActorCritic(state_dim=state_dim, action_dim=2, reward_dim=6, width=int(plan["width"])).to(device)
    model.load_state_dict(copy.deepcopy(initial_state_dict))
    learner = SimplexPPO(model, plan)

    pref_sampler = FaceBalancedPreferenceSampler(
        reward_dim=6,
        mode=arm,
        seed=seed + 10_000,
        interior_prob=float(plan["interior_prob"]),
        vertex_prob=float(plan["vertex_prob"]),
    )

    envs = [make_mo_env("fruit-tree-v0", seed=seed + i) for i in range(streams)]
    observations = np.stack(
        [encode_fruit_tree_node(env.reset(seed=seed + i)[0], tree_depth) for i, env in enumerate(envs)]
    )
    active_prefs = np.stack([pref_sampler.sample() for _ in range(streams)])

    step = 0
    rounds = 0
    episodes_completed = 0
    last_loss: dict[str, float] = {}

    try:
        while step < total_steps:
            ticks = min(fragment_steps, (total_steps - step) // streams)
            batch: dict[str, list[torch.Tensor]] = {
                k: []
                for k in (
                    "states",
                    "preferences",
                    "actions",
                    "logp",
                    "values",
                    "next_values",
                    "rewards",
                    "terminated",
                    "truncated",
                )
            }
            for _ in range(ticks):
                s_t = torch.tensor(observations, dtype=torch.float32, device=device)
                w_t = torch.tensor(active_prefs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    dist = model.distribution(s_t, w_t)
                    actions = dist.sample()
                    logp = dist.log_prob(actions)
                    values = model.value(s_t, w_t)

                next_obs_encoded = np.zeros_like(observations)
                rewards = np.zeros((streams, 6), dtype=np.float32)
                terminations = np.zeros(streams, dtype=bool)
                truncations = np.zeros(streams, dtype=bool)
                actions_np = actions.cpu().numpy()

                for i, env in enumerate(envs):
                    nxt, reward, term, trunc, _ = env.step(int(actions_np[i]))
                    next_obs_encoded[i] = encode_fruit_tree_node(nxt, tree_depth)
                    rewards[i] = np.asarray(reward, dtype=np.float32)
                    terminations[i] = bool(term)
                    truncations[i] = bool(trunc)

                with torch.no_grad():
                    next_values = model.value(torch.tensor(next_obs_encoded, dtype=torch.float32, device=device), w_t)

                batch["states"].append(s_t)
                batch["preferences"].append(w_t)
                batch["actions"].append(actions)
                batch["logp"].append(logp)
                batch["values"].append(values)
                batch["next_values"].append(next_values)
                batch["rewards"].append(torch.tensor(rewards, dtype=torch.float32, device=device))
                batch["terminated"].append(torch.tensor(terminations, dtype=torch.bool, device=device))
                batch["truncated"].append(torch.tensor(truncations, dtype=torch.bool, device=device))

                observations = next_obs_encoded
                step += streams
                for i, env in enumerate(envs):
                    if terminations[i] or truncations[i]:
                        episodes_completed += 1
                        reset_obs, _ = env.reset()
                        observations[i] = encode_fruit_tree_node(reset_obs, tree_depth)
                        active_prefs[i] = pref_sampler.sample()

            last_loss = learner.update({k: torch.stack(v) for k, v in batch.items()}, step)
            rounds += 1
    finally:
        for env in envs:
            env.close()

    train_wall = time.monotonic() - started
    eval_started = time.monotonic()
    model.eval()
    evaluation = evaluate_policy_on_fruit_tree(
        model=model,
        splits=splits,
        oracle_leaves=oracle_leaves,
        gamma=gamma,
        tree_depth=tree_depth,
        device=device,
    )
    eval_wall = time.monotonic() - eval_started

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "arm": arm, "seed": seed, "plan": plan},
            artifact_dir / f"{arm}_seed{seed}.pt",
        )

    return {
        "arm": arm,
        "seed": seed,
        "training_interactions": step,
        "rounds": rounds,
        "actor_updates": learner.actor_updates,
        "critic_updates": learner.critic_updates,
        "episodes_completed": episodes_completed,
        "train_wall_seconds": round(train_wall, 3),
        "eval_wall_seconds": round(eval_wall, 3),
        "total_wall_seconds": round(train_wall + eval_wall, 3),
        "gpu_hours_equivalent": round((train_wall + eval_wall) / 3600.0, 6),
        "final_loss": last_loss,
        "preference_exposure": pref_sampler.exposure_summary(),
        "evaluation_steps": evaluation["evaluation_steps"],
        "modes": evaluation["modes"],
    }


def summarize_paired_metric(
    values_fb: list[float],
    values_unif: list[float],
    seed: int = 20269001,
) -> dict[str, Any]:
    arr_fb = np.asarray(values_fb, dtype=np.float64)
    arr_unif = np.asarray(values_unif, dtype=np.float64)
    diff = arr_fb - arr_unif
    fb_mean, fb_lo, fb_hi = bootstrap_ci(arr_fb, n_boot=5000, seed=seed)
    unif_mean, unif_lo, unif_hi = bootstrap_ci(arr_unif, n_boot=5000, seed=seed + 1)
    diff_mean, diff_lo, diff_hi = bootstrap_ci(diff, n_boot=5000, seed=seed + 2)
    perm = paired_permutation_test(arr_fb, arr_unif, seed=seed + 3)
    dz = float(perm["effect_size_dz"])
    # Approximate two-sided paired t-test sample size for 80% and 90% power at alpha=0.05.
    abs_dz = abs(dz)
    if abs_dz > 1e-6 and math.isfinite(abs_dz):
        n_power_80 = math.ceil(((1.96 + 0.842) / abs_dz) ** 2 + 2)
        n_power_90 = math.ceil(((1.96 + 1.282) / abs_dz) ** 2 + 2)
    else:
        n_power_80 = -1
        n_power_90 = -1
    return {
        "uniform": {"mean": unif_mean, "ci95": [unif_lo, unif_hi], "per_seed": arr_unif.tolist()},
        "fixed_face_balanced": {"mean": fb_mean, "ci95": [fb_lo, fb_hi], "per_seed": arr_fb.tolist()},
        "paired_diff_fb_minus_unif": {
            "mean": diff_mean,
            "ci95": [diff_lo, diff_hi],
            "effect_size_dz": dz,
            "p_value": float(perm["p_value"]),
            "test_mode": perm["mode"],
            "n_seeds_for_80pct_power": n_power_80,
            "n_seeds_for_90pct_power": n_power_90,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20268001 + i for i in range(10)])
    parser.add_argument("--training-interactions", type=int, default=153600)
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--fragment-steps", type=int, default=48)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--interior-prob", type=float, default=0.50)
    parser.add_argument("--vertex-prob", type=float, default=0.25)
    parser.add_argument("--eval-split-seed", type=int, default=20268101)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact-dir", default="artifacts/week1-fruit-tree-pilot-v1")
    parser.add_argument("--output", default="research/evidence/week1_fruit_tree_pilot_v1.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()

    plan: dict[str, Any] = {
        "env_id": "fruit-tree-v0",
        "tree_depth": 6,
        "reward_dim": 6,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "training_interactions": args.training_interactions,
        "streams": args.streams,
        "fragment_steps": args.fragment_steps,
        "width": args.width,
        "epochs": 4,
        "minibatches": 4,
        "critic_steps_per_actor": 2,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "adam_eps": 1e-5,
        "clip": 0.2,
        "max_grad_norm": 0.5,
        "popart_rate": 0.01,
        "popart_std_floor": 1e-3,
        "entropy_start": math.log(2),
        "entropy_end": 0.05,
        "entropy_initial_multiplier": 0.02,
        "entropy_damping": 0.05,
        "entropy_multiplier_lr": 1e-4,
        "hindsight_relabeling": False,
        "interior_prob": args.interior_prob,
        "vertex_prob": args.vertex_prob,
        "eval_epsilon": args.epsilon,
        "eval_split_seed": args.eval_split_seed,
    }

    oracle = extract_fruit_tree_oracle(tree_depth=6, gamma=0.99)
    oracle_leaves = oracle["discounted_leaf_returns"]
    splits = build_fruit_tree_eval_splits(seed=args.eval_split_seed, epsilon=args.epsilon)
    artifact_dir = Path(args.artifact_dir)

    runs_by_arm: dict[str, list[dict[str, Any]]] = {"uniform": [], "fixed_face_balanced": []}
    paired_apm_fb_vs_unif: dict[str, list[float]] = {"argmax": [], "sampled": []}
    paired_apm_unif_vs_fb: dict[str, list[float]] = {"argmax": [], "sampled": []}

    for seed in args.seeds:
        # Paired identical initialization for both arms on the same seed.
        seed_all(seed)
        init_model = ActorCritic(state_dim=127, action_dim=2, reward_dim=6, width=args.width)
        init_state = copy.deepcopy(init_model.state_dict())

        unif_run = train_single_run(
            arm="uniform",
            seed=seed,
            initial_state_dict=init_state,
            plan=plan,
            splits=splits,
            oracle_leaves=oracle_leaves,
            device=args.device,
            artifact_dir=artifact_dir,
        )
        fb_run = train_single_run(
            arm="fixed_face_balanced",
            seed=seed,
            initial_state_dict=init_state,
            plan=plan,
            splits=splits,
            oracle_leaves=oracle_leaves,
            device=args.device,
            artifact_dir=artifact_dir,
        )
        runs_by_arm["uniform"].append(unif_run)
        runs_by_arm["fixed_face_balanced"].append(fb_run)

        for mode in ("argmax", "sampled"):
            unif_ret = np.asarray(unif_run["modes"][mode]["mean_returns_combined"], dtype=np.float64)
            fb_ret = np.asarray(fb_run["modes"][mode]["mean_returns_combined"], dtype=np.float64)
            paired_apm_fb_vs_unif[mode].append(
                compute_abandoned_preference_mass(
                    fb_ret, unif_ret, splits["combined"], tau=0.10, sign_guard="magnitude"
                )["apm"]
            )
            paired_apm_unif_vs_fb[mode].append(
                compute_abandoned_preference_mass(
                    unif_ret, fb_ret, splits["combined"], tau=0.10, sign_guard="magnitude"
                )["apm"]
            )

        u_fc = unif_run["modes"]["argmax"]["splits"]["face_cap"]["deployed_regret"]["mean_regret"]
        f_fc = fb_run["modes"]["argmax"]["splits"]["face_cap"]["deployed_regret"]["mean_regret"]
        u_vr = unif_run["modes"]["argmax"]["splits"]["vertices"]["deployed_regret"]["mean_regret"]
        f_vr = fb_run["modes"]["argmax"]["splits"]["vertices"]["deployed_regret"]["mean_regret"]
        u_int = unif_run["modes"]["argmax"]["splits"]["interior_uniform"]["deployed_regret"]["mean_deployed_utility"]
        f_int = fb_run["modes"]["argmax"]["splits"]["interior_uniform"]["deployed_regret"]["mean_deployed_utility"]
        print(
            f"[seed={seed}] argmax: face_cap_regret unif={u_fc:.4f} -> fb={f_fc:.4f} | "
            f"vertex_regret unif={u_vr:.4f} -> fb={f_vr:.4f} | "
            f"interior_U unif={u_int:.4f} -> fb={f_int:.4f} | "
            f"wall={unif_run['total_wall_seconds']:.1f}s/{fb_run['total_wall_seconds']:.1f}s",
            flush=True,
        )

    paired_summary: dict[str, Any] = {}
    for mode in ("argmax", "sampled"):
        metrics_table = {
            "face_cap_mean_regret": (
                [
                    r["modes"][mode]["splits"]["face_cap"]["deployed_regret"]["mean_regret"]
                    for r in runs_by_arm["fixed_face_balanced"]
                ],
                [
                    r["modes"][mode]["splits"]["face_cap"]["deployed_regret"]["mean_regret"]
                    for r in runs_by_arm["uniform"]
                ],
            ),
            "vertex_mean_regret": (
                [
                    r["modes"][mode]["splits"]["vertices"]["deployed_regret"]["mean_regret"]
                    for r in runs_by_arm["fixed_face_balanced"]
                ],
                [
                    r["modes"][mode]["splits"]["vertices"]["deployed_regret"]["mean_regret"]
                    for r in runs_by_arm["uniform"]
                ],
            ),
            "vertex_cap_mean_regret": (
                [
                    r["modes"][mode]["splits"]["vertex_cap"]["deployed_regret"]["mean_regret"]
                    for r in runs_by_arm["fixed_face_balanced"]
                ],
                [
                    r["modes"][mode]["splits"]["vertex_cap"]["deployed_regret"]["mean_regret"]
                    for r in runs_by_arm["uniform"]
                ],
            ),
            "edge_cap_mean_regret": (
                [
                    r["modes"][mode]["splits"]["edge_cap"]["deployed_regret"]["mean_regret"]
                    for r in runs_by_arm["fixed_face_balanced"]
                ],
                [
                    r["modes"][mode]["splits"]["edge_cap"]["deployed_regret"]["mean_regret"]
                    for r in runs_by_arm["uniform"]
                ],
            ),
            "face_cap_cvar10_regret": (
                [
                    r["modes"][mode]["splits"]["face_cap"]["deployed_regret"]["cvar_10_regret"]
                    for r in runs_by_arm["fixed_face_balanced"]
                ],
                [
                    r["modes"][mode]["splits"]["face_cap"]["deployed_regret"]["cvar_10_regret"]
                    for r in runs_by_arm["uniform"]
                ],
            ),
            "interior_mean_utility": (
                [
                    r["modes"][mode]["splits"]["interior_uniform"]["deployed_regret"]["mean_deployed_utility"]
                    for r in runs_by_arm["fixed_face_balanced"]
                ],
                [
                    r["modes"][mode]["splits"]["interior_uniform"]["deployed_regret"]["mean_deployed_utility"]
                    for r in runs_by_arm["uniform"]
                ],
            ),
            "interior_mean_regret": (
                [
                    r["modes"][mode]["splits"]["interior_uniform"]["deployed_regret"]["mean_regret"]
                    for r in runs_by_arm["fixed_face_balanced"]
                ],
                [
                    r["modes"][mode]["splits"]["interior_uniform"]["deployed_regret"]["mean_regret"]
                    for r in runs_by_arm["uniform"]
                ],
            ),
            "combined_mean_spearman": (
                [
                    r["modes"][mode]["splits"]["combined"]["controllability"]["mean_spearman"]
                    for r in runs_by_arm["fixed_face_balanced"]
                ],
                [
                    r["modes"][mode]["splits"]["combined"]["controllability"]["mean_spearman"]
                    for r in runs_by_arm["uniform"]
                ],
            ),
            "combined_apm_vs_oracle": (
                [
                    r["modes"][mode]["splits"]["combined"]["apm_magnitude"]["apm"]
                    for r in runs_by_arm["fixed_face_balanced"]
                ],
                [r["modes"][mode]["splits"]["combined"]["apm_magnitude"]["apm"] for r in runs_by_arm["uniform"]],
            ),
            "combined_effective_coverage": (
                [
                    r["modes"][mode]["splits"]["combined"]["effective_coverage"]
                    for r in runs_by_arm["fixed_face_balanced"]
                ],
                [r["modes"][mode]["splits"]["combined"]["effective_coverage"] for r in runs_by_arm["uniform"]],
            ),
            "face_cap_archive_blindness_gap": (
                [
                    r["modes"][mode]["splits"]["face_cap"]["deployed_regret"]["archive_blindness_gap"]
                    for r in runs_by_arm["fixed_face_balanced"]
                ],
                [
                    r["modes"][mode]["splits"]["face_cap"]["deployed_regret"]["archive_blindness_gap"]
                    for r in runs_by_arm["uniform"]
                ],
            ),
        }
        mode_summary = {
            name: summarize_paired_metric(fb_vals, unif_vals, seed=20269001 + i * 10)
            for i, (name, (fb_vals, unif_vals)) in enumerate(metrics_table.items())
        }
        primary_keys = [
            "face_cap_mean_regret",
            "vertex_mean_regret",
            "vertex_cap_mean_regret",
            "edge_cap_mean_regret",
            "interior_mean_utility",
        ]
        holm_p = holm_adjust(np.array([mode_summary[k]["paired_diff_fb_minus_unif"]["p_value"] for k in primary_keys]))
        for key, adj_p in zip(primary_keys, holm_p, strict=True):
            mode_summary[key]["paired_diff_fb_minus_unif"]["holm_adjusted_p_value"] = float(adj_p)

        unif_int_u = mode_summary["interior_mean_utility"]["uniform"]["mean"]
        fb_int_u = mode_summary["interior_mean_utility"]["fixed_face_balanced"]["mean"]
        unif_fc_r = mode_summary["face_cap_mean_regret"]["uniform"]["mean"]
        fb_fc_r = mode_summary["face_cap_mean_regret"]["fixed_face_balanced"]["mean"]
        mode_summary["derived_gate_ratios"] = {
            "face_cap_regret_relative_reduction": float((unif_fc_r - fb_fc_r) / max(1e-12, unif_fc_r)),
            "interior_utility_retention_ratio": float(fb_int_u / max(1e-12, unif_int_u)),
            "interior_utility_relative_change": float((fb_int_u - unif_int_u) / max(1e-12, abs(unif_int_u))),
            "mean_paired_apm_fb_vs_unif": float(np.mean(paired_apm_fb_vs_unif[mode])),
            "mean_paired_apm_unif_vs_fb": float(np.mean(paired_apm_unif_vs_fb[mode])),
        }
        paired_summary[mode] = mode_summary

    # Strip heavy per-preference return arrays from compact versioned JSON evidence while keeping full metrics.
    compact_runs: dict[str, list[dict[str, Any]]] = {}
    for arm, run_list in runs_by_arm.items():
        compact_runs[arm] = []
        for r in run_list:
            rc = {k: v for k, v in r.items() if k != "modes"}
            rc["modes"] = {m: {"splits": mdata["splits"]} for m, mdata in r["modes"].items()}
            compact_runs[arm].append(rc)

    all_walls = [r["total_wall_seconds"] for arm_runs in runs_by_arm.values() for r in arm_runs]
    report = {
        "schema_version": 1,
        "study_id": "week1-fruit-tree-pilot-v1",
        "scope": (
            "Week-1 Step-2 paired-seed pilot (Uniform vs Fixed Face-Balanced, no hindsight relabeling) "
            "on fruit-tree-v0 (d=6, exact discounted oracle)"
        ),
        "final_test_steps": 0,
        "total_wall_seconds": round(time.monotonic() - started, 3),
        "timing_budget_check": {
            "mean_seconds_per_run": round(float(np.mean(all_walls)), 3),
            "max_seconds_per_run": round(float(np.max(all_walls)), 3),
            "max_gpu_hours_per_run": round(float(np.max(all_walls)) / 3600.0, 6),
            "under_2_gpu_hours_limit": bool(float(np.max(all_walls)) / 3600.0 <= 2.0),
            "device": args.device,
        },
        "oracle_verification": {
            "env_id": oracle["env_id"],
            "tree_depth": oracle["tree_depth"],
            "reward_dim": oracle["reward_dim"],
            "gamma": oracle["gamma"],
            "num_leaves": oracle["num_leaves"],
            "verified_max_abs_diff_vs_env_pareto_front": oracle["verified_max_abs_diff_vs_env_pareto_front"],
        },
        "plan": plan,
        "seeds": args.seeds,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "paired_summary": paired_summary,
        "runs": compact_runs,
        "provenance": collect_provenance(vars(args)),
    }
    output_path = Path(args.output)
    write_json(output_path, report)
    print(
        f"Wrote {output_path} | total_wall={report['total_wall_seconds']}s | "
        f"max_run={report['timing_budget_check']['max_seconds_per_run']}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

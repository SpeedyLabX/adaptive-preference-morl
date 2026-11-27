#!/usr/bin/env python3
"""Stage-1A CPU exact-oracle suite runner (fruit-tree-v0 d in {3,4,5,6} + deep-sea-treasure-v0 d=2).

Reads the frozen Gate-1B-passed protocol (configs/protocols/stage1_simplex_face_exposure_v1.json),
fails closed if execution_authorized is not True, executes all 146 paired-seed CPU runs with
matched initializations and matched budgets (zero hindsight relabeling), freezes & SHA-256 hashes
all final-budget checkpoints under artifacts/stage1a-cpu-suite-v1/, and evaluates ONLY on the
development splits (final_test_steps = 0).

Computes:
- Primary endpoint: argmax deployed non-vertex face-cap mean regret (exact vertices excluded).
- Shifted interior-utility retention with strict 95% paired bootstrap CI lower bound >= -0.015.
- Contrasts: fixed_face_balanced vs uniform, fixed_face_balanced vs dirichlet_sparse(0.25),
  and dirichlet_sparse(0.25) vs uniform.
- Frozen within-environment d-scaling OLS slope test across fruit-tree-v0 d in {3, 4, 5, 6}
  (95% stratified seed-bootstrap CI + one-sided d-label permutation p-value).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.envs.dst_oracle import encode_dst_state, extract_dst_oracle  # noqa: E402
from src.envs.fruit_tree_oracle import encode_fruit_tree_node, exact_oracle_utility, extract_fruit_tree_oracle  # noqa: E402
from src.envs.wrapper import make_mo_env  # noqa: E402
from src.evaluation.metrics import compute_expected_utility  # noqa: E402
from src.evaluation.simplex_metrics import (  # noqa: E402
    FaceBalancedPreferenceSampler,
    compute_checkpoint_metrics_bundle,
    compute_shifted_interior_retention,
    sample_edge_cap_preferences,
    sample_uniform_simplex,
    sample_vertex_cap_preferences,
)
from src.evaluation.stats import bootstrap_ci, paired_permutation_test  # noqa: E402
from src.models.face_exposure_ppo import SimplexPPO  # noqa: E402
from src.models.onpolicy import ActorCritic  # noqa: E402
from src.utils.provenance import collect_provenance, write_json  # noqa: E402


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dev_splits(
    reward_dim: int,
    dev_seed: int,
    epsilon: float,
    vertex_cap_total: int,
    edge_cap_total: int,
    interior_count: int,
) -> dict[str, np.ndarray]:
    """Generate deterministic development splits (never touches final_test_seed)."""
    rng = np.random.default_rng(dev_seed)
    vertices = np.eye(reward_dim, dtype=np.float32)
    per_vertex = vertex_cap_total // reward_dim
    vertex_cap = sample_vertex_cap_preferences(reward_dim, per_vertex=per_vertex, epsilon=epsilon, rng=rng)
    if edge_cap_total > 0 and reward_dim >= 3:
        n_edges = reward_dim * (reward_dim - 1) // 2
        if edge_cap_total % n_edges == 0:
            per_edge = edge_cap_total // n_edges
            edge_cap = sample_edge_cap_preferences(reward_dim, per_edge=per_edge, epsilon=epsilon, rng=rng)
        else:
            per_edge = math.ceil(edge_cap_total / n_edges)
            raw_ec = sample_edge_cap_preferences(reward_dim, per_edge=per_edge, epsilon=epsilon, rng=rng)
            chosen = rng.choice(len(raw_ec), size=edge_cap_total, replace=False)
            chosen.sort()
            edge_cap = raw_ec[chosen]
        non_vertex_face_cap = np.vstack([vertex_cap, edge_cap])
    else:
        edge_cap = np.empty((0, reward_dim), dtype=np.float32)
        non_vertex_face_cap = vertex_cap.copy()

    interior_uniform = sample_uniform_simplex(reward_dim, count=interior_count, rng=rng)
    face_cap_incl_vertices = np.vstack([vertices, non_vertex_face_cap])
    combined = np.vstack([interior_uniform, face_cap_incl_vertices])
    return {
        "vertices": vertices,
        "vertex_cap": vertex_cap,
        "edge_cap": edge_cap,
        "non_vertex_face_cap": non_vertex_face_cap,
        "face_cap_incl_vertices": face_cap_incl_vertices,
        "interior_uniform": interior_uniform,
        "combined": combined,
    }


@torch.no_grad()
def evaluate_cpu_policy(
    model: ActorCritic,
    env_id: str,
    reward_dim: int,
    objective_indices: list[int] | None,
    splits: dict[str, np.ndarray],
    oracle_front: np.ndarray,
    reference_point: np.ndarray,
    gamma: float,
    sampled_rollouts: int = 5,
    eval_seed: int = 20269951,
) -> dict[str, Any]:
    env = make_mo_env(env_id)
    max_steps = 6 if env_id == "fruit-tree-v0" else 100
    n_int = len(splits["interior_uniform"])
    n_vert = len(splits["vertices"])
    n_vcap = len(splits["vertex_cap"])
    n_ecap = len(splits["edge_cap"])
    split_slices = {
        "interior_uniform": slice(0, n_int),
        "vertices": slice(n_int, n_int + n_vert),
        "vertex_cap": slice(n_int + n_vert, n_int + n_vert + n_vcap),
        "edge_cap": slice(n_int + n_vert + n_vcap, n_int + n_vert + n_vcap + n_ecap),
        "non_vertex_face_cap": slice(n_int + n_vert, n_int + n_vert + n_vcap + n_ecap),
        "face_cap_incl_vertices": slice(n_int, n_int + n_vert + n_vcap + n_ecap),
        "combined": slice(0, n_int + n_vert + n_vcap + n_ecap),
    }

    combined_prefs = splits["combined"]
    opt_u_all, _ = exact_oracle_utility(combined_prefs, oracle_front)
    modes_out: dict[str, Any] = {}
    total_eval_steps = 0

    try:
        for mode in ("argmax", "sampled"):
            n_reps = 1 if mode == "argmax" else sampled_rollouts
            mean_returns = np.zeros((len(combined_prefs), reward_dim), dtype=np.float64)
            opt_hits = np.zeros(len(combined_prefs), dtype=np.float64)
            for idx, pref in enumerate(combined_prefs):
                w_t = torch.tensor(pref, dtype=torch.float32).unsqueeze(0)
                ep_returns = np.zeros((n_reps, reward_dim), dtype=np.float64)
                hits = 0
                for rep in range(n_reps):
                    if mode == "sampled":
                        torch.manual_seed(eval_seed + idx * 100 + rep)
                    obs, _ = env.reset(seed=0)
                    discounted = np.zeros(reward_dim, dtype=np.float64)
                    discount = 1.0
                    for _ in range(max_steps):
                        enc = encode_fruit_tree_node(obs, 6) if env_id == "fruit-tree-v0" else encode_dst_state(obs)
                        s_t = torch.tensor(enc, dtype=torch.float32).unsqueeze(0)
                        dist = model.distribution(s_t, w_t)
                        action = int(dist.logits.argmax().item() if mode == "argmax" else dist.sample().item())
                        obs, reward, terminated, truncated, _ = env.step(action)
                        r_arr = np.asarray(reward, dtype=np.float64)
                        if objective_indices is not None:
                            r_arr = r_arr[objective_indices]
                        discounted += discount * r_arr
                        discount *= gamma
                        total_eval_steps += 1
                        if terminated or truncated:
                            break
                    ep_returns[rep] = discounted
                    if float(np.dot(pref, discounted)) >= float(opt_u_all[idx]) - 1e-4:
                        hits += 1
                mean_returns[idx] = ep_returns.mean(axis=0)
                opt_hits[idx] = hits / float(n_reps)

            split_metrics: dict[str, Any] = {}
            for split_name, sl in split_slices.items():
                sub_pref = splits[split_name]
                if len(sub_pref) == 0:
                    continue
                sub_ret = mean_returns[sl]
                bundle = compute_checkpoint_metrics_bundle(
                    deployed_returns=sub_ret,
                    preferences=sub_pref,
                    reference_front=oracle_front,
                    reference_point=reference_point,
                    tau=0.10,
                )
                bundle["expected_utility"] = compute_expected_utility(sub_ret, sub_pref)
                bundle["exact_optimal_rate"] = float(np.mean(opt_hits[sl]))
                split_metrics[split_name] = bundle

            modes_out[mode] = {
                "splits": split_metrics,
                "interior_returns": mean_returns[split_slices["interior_uniform"]].tolist(),
            }
    finally:
        env.close()

    return {"modes": modes_out, "evaluation_steps": total_eval_steps}


def run_paired_seed_job(job: dict[str, Any]) -> dict[str, Any]:
    """Train all arms for one paired seed sequentially inside one worker process (matched CPU & init)."""
    env_id: str = job["env_id"]
    group_id: str = job["group_id"]
    reward_dim: int = job["reward_dim"]
    objective_indices: list[int] | None = job["objective_indices"]
    seed: int = job["seed"]
    arms: list[str] = job["arms"]
    plan: dict[str, Any] = job["plan"]
    splits: dict[str, np.ndarray] = job["splits"]
    oracle_front: np.ndarray = job["oracle_front"]
    reference_point: np.ndarray = job["reference_point"]
    artifact_root = Path(job["artifact_root"])

    state_dim = 127 if env_id == "fruit-tree-v0" else 121
    action_dim = 2 if env_id == "fruit-tree-v0" else 4
    width = int(plan["width"])
    streams = int(plan["streams"])
    fragment_steps = int(plan["fragment_steps"])
    total_steps = int(plan["training_interactions"])
    gamma = float(plan["gamma"])

    # Shared paired initial weights across all arms for this seed.
    _seed_all(seed)
    init_model = ActorCritic(state_dim=state_dim, action_dim=action_dim, reward_dim=reward_dim, width=width)
    init_state = copy.deepcopy(init_model.state_dict())

    arm_outputs: dict[str, dict[str, Any]] = {}
    for arm in arms:
        t0 = time.monotonic()
        _seed_all(seed)
        model = ActorCritic(state_dim=state_dim, action_dim=action_dim, reward_dim=reward_dim, width=width)
        model.load_state_dict(copy.deepcopy(init_state))
        learner = SimplexPPO(model, plan)

        interior_prob = 0.50
        vertex_prob = 0.50 if reward_dim == 2 else 0.25
        sampler = FaceBalancedPreferenceSampler(
            reward_dim=reward_dim,
            mode=arm,
            seed=seed + 10_000,
            interior_prob=interior_prob,
            vertex_prob=vertex_prob,
            dirichlet_alpha=0.25,
        )

        envs = [make_mo_env(env_id, seed=seed + i) for i in range(streams)]
        enc_fn = (lambda o: encode_fruit_tree_node(o, 6)) if env_id == "fruit-tree-v0" else encode_dst_state
        observations = np.stack([enc_fn(env.reset(seed=seed + i)[0]) for i, env in enumerate(envs)])
        active_prefs = np.stack([sampler.sample() for _ in range(streams)])

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
                    s_t = torch.tensor(observations, dtype=torch.float32)
                    w_t = torch.tensor(active_prefs, dtype=torch.float32)
                    with torch.no_grad():
                        dist = model.distribution(s_t, w_t)
                        actions = dist.sample()
                        logp = dist.log_prob(actions)
                        values = model.value(s_t, w_t)

                    next_obs = np.zeros_like(observations)
                    rewards = np.zeros((streams, reward_dim), dtype=np.float32)
                    terms = np.zeros(streams, dtype=bool)
                    truncs = np.zeros(streams, dtype=bool)
                    acts_np = actions.numpy()
                    for i, env in enumerate(envs):
                        nxt, r, tm, tr, _ = env.step(int(acts_np[i]))
                        next_obs[i] = enc_fn(nxt)
                        rv = np.asarray(r, dtype=np.float32)
                        rewards[i] = rv[objective_indices] if objective_indices is not None else rv
                        terms[i] = bool(tm)
                        truncs[i] = bool(tr)

                    with torch.no_grad():
                        next_values = model.value(torch.tensor(next_obs, dtype=torch.float32), w_t)

                    batch["states"].append(s_t)
                    batch["preferences"].append(w_t)
                    batch["actions"].append(actions)
                    batch["logp"].append(logp)
                    batch["values"].append(values)
                    batch["next_values"].append(next_values)
                    batch["rewards"].append(torch.tensor(rewards, dtype=torch.float32))
                    batch["terminated"].append(torch.tensor(terms, dtype=torch.bool))
                    batch["truncated"].append(torch.tensor(truncs, dtype=torch.bool))

                    observations = next_obs
                    step += streams
                    for i, env in enumerate(envs):
                        if terms[i] or truncs[i]:
                            episodes_completed += 1
                            rst, _ = env.reset()
                            observations[i] = enc_fn(rst)
                            active_prefs[i] = sampler.sample()

                last_loss = learner.update({k: torch.stack(v) for k, v in batch.items()}, step)
                rounds += 1
        finally:
            for env in envs:
                env.close()

        train_wall = time.monotonic() - t0
        ckpt_dir = artifact_root / group_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{arm}_seed{seed}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "arm": arm,
                "seed": seed,
                "group_id": group_id,
                "env_id": env_id,
                "reward_dim": reward_dim,
                "objective_indices": objective_indices,
                "plan": plan,
            },
            ckpt_path,
        )
        ckpt_sha = _file_sha256(ckpt_path)

        t_eval = time.monotonic()
        model.eval()
        evaluation = evaluate_cpu_policy(
            model=model,
            env_id=env_id,
            reward_dim=reward_dim,
            objective_indices=objective_indices,
            splits=splits,
            oracle_front=oracle_front,
            reference_point=reference_point,
            gamma=gamma,
        )
        eval_wall = time.monotonic() - t_eval

        arm_outputs[arm] = {
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
            "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
            "checkpoint_sha256": ckpt_sha,
            "final_loss": last_loss,
            "preference_exposure": sampler.exposure_summary(),
            "development_evaluation_steps": evaluation["evaluation_steps"],
            "modes": evaluation["modes"],
        }

    return {"group_id": group_id, "seed": seed, "arms": arm_outputs}


def summarize_paired_contrast(
    vals_a: list[float],
    vals_b: list[float],
    name_a: str,
    name_b: str,
    seed: int,
) -> dict[str, Any]:
    arr_a = np.asarray(vals_a, dtype=np.float64)
    arr_b = np.asarray(vals_b, dtype=np.float64)
    diff = arr_a - arr_b
    a_mean, a_lo, a_hi = bootstrap_ci(arr_a, n_boot=10_000, seed=seed)
    b_mean, b_lo, b_hi = bootstrap_ci(arr_b, n_boot=10_000, seed=seed + 1)
    d_mean, d_lo, d_hi = bootstrap_ci(diff, n_boot=10_000, seed=seed + 2)
    perm = paired_permutation_test(arr_a, arr_b, seed=seed + 3)
    rel_change = float((a_mean - b_mean) / max(1e-12, abs(b_mean)))
    return {
        name_b: {"mean": b_mean, "ci95": [b_lo, b_hi], "per_seed": arr_b.tolist()},
        name_a: {"mean": a_mean, "ci95": [a_lo, a_hi], "per_seed": arr_a.tolist()},
        "paired_diff_a_minus_b": {
            "mean": d_mean,
            "ci95": [d_lo, d_hi],
            "relative_change": rel_change,
            "effect_size_dz": float(perm["effect_size_dz"]),
            "p_value": float(perm["p_value"]),
            "test_mode": perm["mode"],
        },
    }


def compute_d_scaling_ols_test(
    per_d_values: dict[int, list[float]],
    seed: int = 20269971,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
) -> dict[str, Any]:
    """Frozen Gate-1B OLS slope test of seed-level regret reduction regressed on d in {3, 4, 5, 6}."""
    dims = sorted(per_d_values)
    xs = np.concatenate([np.full(len(per_d_values[d]), float(d), dtype=np.float64) for d in dims])
    ys = np.concatenate([np.asarray(per_d_values[d], dtype=np.float64) for d in dims])

    def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
        xc = x - x.mean()
        yc = y - y.mean()
        denom = float(np.dot(xc, xc))
        return float(np.dot(xc, yc) / denom) if denom > 0.0 else 0.0

    observed_slope = _ols_slope(xs, ys)
    rng = np.random.default_rng(seed)

    # Stratified seed bootstrap within each dimension d.
    boot_slopes = np.empty(n_boot, dtype=np.float64)
    arrs_by_d = {d: np.asarray(per_d_values[d], dtype=np.float64) for d in dims}
    for b_idx in range(n_boot):
        yb = np.concatenate([rng.choice(arrs_by_d[d], size=len(arrs_by_d[d]), replace=True) for d in dims])
        boot_slopes[b_idx] = _ols_slope(xs, yb)
    ci_lo, ci_hi = float(np.quantile(boot_slopes, 0.025)), float(np.quantile(boot_slopes, 0.975))

    # One-sided d-label permutation test for H1: beta_d > 0.
    exceedances = 0
    for _ in range(n_perm):
        xp = rng.permutation(xs)
        if _ols_slope(xp, ys) >= observed_slope - 1e-15:
            exceedances += 1
    p_one_sided = (exceedances + 1) / float(n_perm + 1)

    return {
        "dims": dims,
        "per_d_means": {str(d): float(np.mean(per_d_values[d])) for d in dims},
        "per_d_counts": {str(d): len(per_d_values[d]) for d in dims},
        "ols_slope_beta_d": observed_slope,
        "ci95_stratified_bootstrap": [ci_lo, ci_hi],
        "p_value_one_sided_beta_gt_0": float(p_one_sided),
        "n_boot": n_boot,
        "n_perm": n_perm,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/protocols/stage1_simplex_face_exposure_v1.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--artifact-dir", default="artifacts/stage1a-cpu-suite-v1")
    parser.add_argument("--output", default="research/evidence/stage1a_cpu_exact_oracle_dev_v1.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()

    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        not protocol.get("execution_authorized")
        or protocol.get("week1_gate_prerequisites", {}).get("gate1b_verdict") != "PASS"
    ):
        raise RuntimeError("Protocol execution_authorized is not True or Gate-1B PASS missing; refusing to run.")

    artifact_root = REPO_ROOT / args.artifact_dir
    artifact_root.mkdir(parents=True, exist_ok=True)

    # Build group specs: fruit-tree d=6 core, fruit-tree d in {3,4,5} subsets, and DST d=2 sanity.
    benchmarks = {b["env_id"]: b for b in protocol["benchmarks"]}
    ft_core = benchmarks["fruit-tree-v0"]
    dst_bench = benchmarks["deep-sea-treasure-v0"]
    sweep = protocol["fruit_tree_d_subset_scaling_sweep"]

    base_ft_plan = {
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "training_interactions": int(ft_core["training_interactions"]),
        "streams": int(ft_core["streams"]),
        "fragment_steps": int(ft_core["fragment_steps"]),
        "width": int(ft_core["width"]),
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
    }
    dst_plan = {
        **base_ft_plan,
        "gae_lambda": 0.95,
        "training_interactions": int(dst_bench["training_interactions"]),
        "streams": int(dst_bench["streams"]),
        "fragment_steps": int(dst_bench["fragment_steps"]),
        "width": int(dst_bench["width"]),
        "entropy_start": math.log(4),
    }

    group_meta: dict[str, dict[str, Any]] = {}
    jobs: list[dict[str, Any]] = []

    # 1. Fruit-Tree d in {3, 4, 5, 6}
    for sub in sweep["subsets"]:
        dim = int(sub["reward_dim"])
        obj_idx = list(sub["objective_indices"])
        oracle = extract_fruit_tree_oracle(tree_depth=6, gamma=0.99, objective_indices=obj_idx)
        ref_pt = np.zeros(dim, dtype=np.float64)
        if sub.get("reuses_core_benchmark_runs"):
            group_id = "fruit_tree_d6"
            seeds = list(ft_core["paired_seeds"])
            arms = list(ft_core["arms_included"])
            dev_seed = int(ft_core["evaluation_splits"]["development_seed"])
            comp = ft_core["evaluation_splits"]["primary_non_vertex_face_cap_composition"]
            int_cnt = int(ft_core["evaluation_splits"]["counts"]["interior_uniform"])
            vcap_cnt, ecap_cnt = int(comp["vertex_cap_count"]), int(comp["edge_cap_count"])
        else:
            group_id = f"fruit_tree_d{dim}"
            seeds = list(sub["paired_seeds"])
            arms = list(sweep["arms_included"])
            dev_seed = int(sub["development_seed"])
            nv = sub["non_vertex_face_cap_counts"]
            vcap_cnt, ecap_cnt, int_cnt = int(nv["vertex_cap"]), int(nv["edge_cap"]), int(nv["interior_uniform"])

        splits = build_dev_splits(dim, dev_seed, 0.10, vcap_cnt, ecap_cnt, int_cnt)
        group_meta[group_id] = {
            "env_id": "fruit-tree-v0",
            "reward_dim": dim,
            "objective_indices": obj_idx,
            "arms": arms,
            "seeds": seeds,
            "dev_seed": dev_seed,
            "reference_point": ref_pt,
            "splits": splits,
            "oracle_max_diff": oracle["verified_max_abs_diff_vs_env_pareto_front"],
        }
        for s in seeds:
            jobs.append(
                {
                    "group_id": group_id,
                    "env_id": "fruit-tree-v0",
                    "reward_dim": dim,
                    "objective_indices": obj_idx,
                    "seed": s,
                    "arms": arms,
                    "plan": base_ft_plan,
                    "splits": splits,
                    "oracle_front": oracle["discounted_leaf_returns"],
                    "reference_point": ref_pt,
                    "artifact_root": str(artifact_root),
                }
            )

    # 2. Deep-Sea-Treasure d=2 sanity check
    dst_oracle = extract_dst_oracle(gamma=0.99)
    dst_ref = np.asarray(dst_bench["reference_point"], dtype=np.float64)
    dst_dev_seed = int(dst_bench["evaluation_splits"]["development_seed"])
    dst_comp = dst_bench["evaluation_splits"]["primary_non_vertex_face_cap_composition"]
    dst_int = int(dst_bench["evaluation_splits"]["counts"]["interior_uniform"])
    dst_splits = build_dev_splits(2, dst_dev_seed, 0.10, int(dst_comp["vertex_cap_count"]), 0, dst_int)
    group_meta["dst_d2_sanity"] = {
        "env_id": "deep-sea-treasure-v0",
        "reward_dim": 2,
        "objective_indices": [0, 1],
        "arms": list(dst_bench["arms_included"]),
        "seeds": list(dst_bench["paired_seeds"]),
        "dev_seed": dst_dev_seed,
        "reference_point": dst_ref,
        "splits": dst_splits,
        "oracle_max_diff": dst_oracle["verified_max_abs_diff_vs_env_pareto_front"],
    }
    for s in dst_bench["paired_seeds"]:
        jobs.append(
            {
                "group_id": "dst_d2_sanity",
                "env_id": "deep-sea-treasure-v0",
                "reward_dim": 2,
                "objective_indices": None,
                "seed": s,
                "arms": list(dst_bench["arms_included"]),
                "plan": dst_plan,
                "splits": dst_splits,
                "oracle_front": dst_oracle["discounted_pareto_front"],
                "reference_point": dst_ref,
                "artifact_root": str(artifact_root),
            }
        )

    total_runs = sum(len(j["arms"]) for j in jobs)
    print(
        f"Dispatching {len(jobs)} paired-seed jobs ({total_runs} CPU runs) across {args.workers} workers...", flush=True
    )

    results_by_group: dict[str, dict[int, dict[str, Any]]] = {g: {} for g in group_meta}
    done_jobs = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_paired_seed_job, job) for job in jobs]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            gid, s = res["group_id"], res["seed"]
            results_by_group[gid][s] = res["arms"]
            done_jobs += 1
            u_r = res["arms"]["uniform"]["modes"]["argmax"]["splits"]["non_vertex_face_cap"]["deployed_regret"][
                "mean_regret"
            ]
            fb_r = res["arms"]["fixed_face_balanced"]["modes"]["argmax"]["splits"]["non_vertex_face_cap"][
                "deployed_regret"
            ]["mean_regret"]
            sp_str = ""
            if "dirichlet_sparse" in res["arms"]:
                ds_r = res["arms"]["dirichlet_sparse"]["modes"]["argmax"]["splits"]["non_vertex_face_cap"][
                    "deployed_regret"
                ]["mean_regret"]
                sp_str = f" dir025={ds_r:.4f}"
            print(
                f"[{done_jobs:2d}/{len(jobs)}] {gid} seed={s}: "
                f"argmax non_vertex_regret unif={u_r:.4f}{sp_str} fb={fb_r:.4f}",
                flush=True,
            )

    # Aggregate statistics per group
    group_summaries: dict[str, Any] = {}
    d_scaling_rel_fb_vs_unif: dict[int, list[float]] = {}
    d_scaling_abs_fb_vs_unif: dict[int, list[float]] = {}
    d_scaling_rel_fb_vs_dir: dict[int, list[float]] = {}
    d_scaling_abs_fb_vs_dir: dict[int, list[float]] = {}

    for gid, meta in group_meta.items():
        seeds = meta["seeds"]
        arms = meta["arms"]
        int_prefs = meta["splits"]["interior_uniform"]
        ref_pt = meta["reference_point"]

        def _extract(
            arm_name: str, mode: str, split: str, metric: str, g: str = gid, sd: list[int] = seeds
        ) -> list[float]:
            return [
                results_by_group[g][s][arm_name]["modes"][mode]["splits"][split]["deployed_regret"][metric] for s in sd
            ]

        contrasts: dict[str, Any] = {}
        contrast_pairs = [("fixed_face_balanced", "uniform")]
        if "dirichlet_sparse" in arms:
            contrast_pairs.extend([("fixed_face_balanced", "dirichlet_sparse"), ("dirichlet_sparse", "uniform")])

        for pair_idx, (arm_a, arm_b) in enumerate(contrast_pairs):
            cname = f"{arm_a}_vs_{arm_b}"
            # Shifted interior utility relative change per seed (Fix 1 strict CI >= -0.015).
            shifted_rel_argmax: list[float] = []
            shifted_rel_sampled: list[float] = []
            for s in seeds:
                ret_a = np.asarray(results_by_group[gid][s][arm_a]["modes"]["argmax"]["interior_returns"])
                ret_b = np.asarray(results_by_group[gid][s][arm_b]["modes"]["argmax"]["interior_returns"])
                shifted_rel_argmax.append(
                    compute_shifted_interior_retention(ret_a, ret_b, int_prefs, ref_pt)["shifted_relative_change"]
                )
                ret_as = np.asarray(results_by_group[gid][s][arm_a]["modes"]["sampled"]["interior_returns"])
                ret_bs = np.asarray(results_by_group[gid][s][arm_b]["modes"]["sampled"]["interior_returns"])
                shifted_rel_sampled.append(
                    compute_shifted_interior_retention(ret_as, ret_bs, int_prefs, ref_pt)["shifted_relative_change"]
                )
            ret_mean, ret_lo, ret_hi = bootstrap_ci(
                np.asarray(shifted_rel_argmax), n_boot=10_000, seed=20269800 + pair_idx
            )
            rets_mean, rets_lo, rets_hi = bootstrap_ci(
                np.asarray(shifted_rel_sampled), n_boot=10_000, seed=20269850 + pair_idx
            )

            contrasts[cname] = {
                "argmax_primary_non_vertex_face_cap_regret": summarize_paired_contrast(
                    _extract(arm_a, "argmax", "non_vertex_face_cap", "mean_regret"),
                    _extract(arm_b, "argmax", "non_vertex_face_cap", "mean_regret"),
                    arm_a,
                    arm_b,
                    seed=20269100 + pair_idx * 20,
                ),
                "sampled_non_vertex_face_cap_regret": summarize_paired_contrast(
                    _extract(arm_a, "sampled", "non_vertex_face_cap", "mean_regret"),
                    _extract(arm_b, "sampled", "non_vertex_face_cap", "mean_regret"),
                    arm_a,
                    arm_b,
                    seed=20269200 + pair_idx * 20,
                ),
                "argmax_exact_vertex_regret": summarize_paired_contrast(
                    _extract(arm_a, "argmax", "vertices", "mean_regret"),
                    _extract(arm_b, "argmax", "vertices", "mean_regret"),
                    arm_a,
                    arm_b,
                    seed=20269300 + pair_idx * 20,
                ),
                "argmax_cvar10_non_vertex_regret": summarize_paired_contrast(
                    _extract(arm_a, "argmax", "non_vertex_face_cap", "cvar_10_regret"),
                    _extract(arm_b, "argmax", "non_vertex_face_cap", "cvar_10_regret"),
                    arm_a,
                    arm_b,
                    seed=20269400 + pair_idx * 20,
                ),
                "argmax_shifted_interior_retention": {
                    "mean_relative_change": ret_mean,
                    "ci95": [ret_lo, ret_hi],
                    "passes_strict_ci_ge_minus_0015": bool(ret_lo >= -0.015),
                    "per_seed": shifted_rel_argmax,
                },
                "sampled_shifted_interior_retention": {
                    "mean_relative_change": rets_mean,
                    "ci95": [rets_lo, rets_hi],
                    "passes_strict_ci_ge_minus_0015": bool(rets_lo >= -0.015),
                    "per_seed": shifted_rel_sampled,
                },
            }

        if gid.startswith("fruit_tree_d"):
            dim = int(meta["reward_dim"])
            u_vals = _extract("uniform", "argmax", "non_vertex_face_cap", "mean_regret")
            f_vals = _extract("fixed_face_balanced", "argmax", "non_vertex_face_cap", "mean_regret")
            d_vals = _extract("dirichlet_sparse", "argmax", "non_vertex_face_cap", "mean_regret")
            d_scaling_rel_fb_vs_unif[dim] = [(u - f) / max(1e-12, u) for u, f in zip(u_vals, f_vals, strict=True)]
            d_scaling_abs_fb_vs_unif[dim] = [u - f for u, f in zip(u_vals, f_vals, strict=True)]
            d_scaling_rel_fb_vs_dir[dim] = [(ds - f) / max(1e-12, ds) for ds, f in zip(d_vals, f_vals, strict=True)]
            d_scaling_abs_fb_vs_dir[dim] = [ds - f for ds, f in zip(d_vals, f_vals, strict=True)]

        # Compact per-run records (drop per-preference interior return vectors to keep JSON lean).
        compact_runs: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
        for s in seeds:
            for a in arms:
                rec = {k: v for k, v in results_by_group[gid][s][a].items() if k != "modes"}
                rec["modes"] = {
                    m: {"splits": results_by_group[gid][s][a]["modes"][m]["splits"]} for m in ("argmax", "sampled")
                }
                compact_runs[a].append(rec)

        group_summaries[gid] = {
            "env_id": meta["env_id"],
            "reward_dim": meta["reward_dim"],
            "objective_indices": meta["objective_indices"],
            "arms": arms,
            "seeds": seeds,
            "development_split_seed": meta["dev_seed"],
            "oracle_verified_max_abs_diff": meta["oracle_max_diff"],
            "contrasts": contrasts,
            "runs": compact_runs,
        }

    d_scaling_analysis = {
        "fixed_face_balanced_vs_uniform_relative_reduction": compute_d_scaling_ols_test(
            d_scaling_rel_fb_vs_unif, seed=20269971
        ),
        "fixed_face_balanced_vs_uniform_absolute_reduction": compute_d_scaling_ols_test(
            d_scaling_abs_fb_vs_unif, seed=20269972
        ),
        "fixed_face_balanced_vs_dirichlet_sparse_relative_reduction": compute_d_scaling_ols_test(
            d_scaling_rel_fb_vs_dir, seed=20269973
        ),
        "fixed_face_balanced_vs_dirichlet_sparse_absolute_reduction": compute_d_scaling_ols_test(
            d_scaling_abs_fb_vs_dir, seed=20269974
        ),
    }

    all_walls = [
        results_by_group[g][s][a]["total_wall_seconds"]
        for g, meta in group_meta.items()
        for s in meta["seeds"]
        for a in meta["arms"]
    ]
    total_dev_steps = sum(
        results_by_group[g][s][a]["development_evaluation_steps"]
        for g, meta in group_meta.items()
        for s in meta["seeds"]
        for a in meta["arms"]
    )

    report = {
        "schema_version": 1,
        "study_id": "stage1a-cpu-exact-oracle-dev-v1",
        "protocol_path": args.protocol,
        "protocol_sha256": _file_sha256(protocol_path),
        "scope": (
            "Stage-1A CPU exact-oracle suite (fruit-tree-v0 d in {3,4,5,6} + deep-sea-treasure-v0 d=2 sanity), "
            "evaluated strictly on development splits; final_test_steps = 0."
        ),
        "final_test_steps": 0,
        "total_development_evaluation_steps": total_dev_steps,
        "total_runs": total_runs,
        "total_wall_seconds": round(time.monotonic() - started, 3),
        "per_run_timing": {
            "mean_seconds": round(float(np.mean(all_walls)), 3),
            "max_seconds": round(float(np.max(all_walls)), 3),
            "sum_cpu_hours": round(float(np.sum(all_walls)) / 3600.0, 4),
        },
        "d_scaling_ols_analysis": d_scaling_analysis,
        "groups": group_summaries,
        "provenance": collect_provenance(vars(args)),
    }
    out_path = Path(args.output)
    write_json(out_path, report)
    print(
        f"Wrote {out_path} | runs={total_runs} | wall={report['total_wall_seconds']}s | "
        f"sum_cpu_h={report['per_run_timing']['sum_cpu_hours']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

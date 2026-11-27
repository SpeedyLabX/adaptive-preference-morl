#!/usr/bin/env python3
"""Stage-1A GPU job entry point for hei-research (21 vertex-specialist anchors + Minecart/Lunar arms).

Reads job parameters from hei-research environment variables (HEI_DATASET, HEI_SEED, HEI_DEVICE,
HEI_OUT, HEI_RUN_ID, HEI_COMPATIBILITY) or CLI flags, validates against the frozen Gate-1B-passed
protocol (configs/protocols/stage1_simplex_face_exposure_v1.json), trains a single fixed-budget
4M-interaction SimplexPPO run (zero hindsight relabeling), saves and SHA-256 hashes the final-budget
checkpoint, writes a schema-1 hei-research result.json, and strictly keeps final_test_steps = 0.

HEI_DATASET format:
  - Anchor run: "<env_id>|anchor|vertex=<idx>" (e.g., "minecart-v0|anchor|vertex=0")
  - Arm run:    "<env_id>|arm|<arm_id>"       (e.g., "mo-lunar-lander-v3|arm|fixed_face_balanced")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.envs.wrapper import make_mo_env  # noqa: E402
from src.evaluation.simplex_metrics import FaceBalancedPreferenceSampler  # noqa: E402
from src.models.face_exposure_ppo import SimplexPPO  # noqa: E402
from src.models.onpolicy import ActorCritic  # noqa: E402


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_sha256(model: ActorCritic) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def parse_dataset_spec(dataset_spec: str) -> tuple[str, str, str]:
    parts = dataset_spec.strip().split("|")
    if len(parts) != 3 or parts[1] not in {"anchor", "arm"}:
        raise ValueError(f"Expected HEI_DATASET '<env_id>|<anchor|arm>|<spec>', got: {dataset_spec!r}")
    return parts[0], parts[1], parts[2]


@torch.no_grad()
def evaluate_vertex_anchor(
    model: ActorCritic,
    env_id: str,
    vertex_idx: int,
    reward_dim: int,
    scale: np.ndarray,
    gamma: float,
    device: str,
    eval_seeds: list[int],
) -> dict[str, Any]:
    """Quick deterministic development evaluation of a vertex specialist at its own vertex e_i (final_test_steps=0)."""
    env = make_mo_env(env_id)
    pref = np.zeros(reward_dim, dtype=np.float32)
    pref[vertex_idx] = 1.0
    w_t = torch.tensor(pref, dtype=torch.float32, device=device).unsqueeze(0)
    model.eval()
    modes_out: dict[str, Any] = {}
    eval_steps = 0
    try:
        for mode in ("argmax", "sampled"):
            raw_list: list[np.ndarray] = []
            scaled_list: list[np.ndarray] = []
            for rep_idx, s in enumerate(eval_seeds):
                torch.manual_seed(s + vertex_idx * 100 + rep_idx)
                np.random.seed(s + vertex_idx * 100 + rep_idx)
                obs, _ = env.reset(seed=s)
                raw = np.zeros(reward_dim, dtype=np.float64)
                disc = np.zeros(reward_dim, dtype=np.float64)
                discount = 1.0
                for _ in range(1000):
                    s_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    dist = model.distribution(s_t, w_t)
                    act = int(dist.logits.argmax().item() if mode == "argmax" else dist.sample().item())
                    obs, r, tm, tr, _ = env.step(act)
                    rv = np.asarray(r, dtype=np.float64)
                    raw += rv
                    disc += discount * rv
                    discount *= gamma
                    eval_steps += 1
                    if tm or tr:
                        break
                raw_list.append(raw)
                scaled_list.append(disc * scale)
            mean_scaled = np.mean(scaled_list, axis=0)
            mean_raw = np.mean(raw_list, axis=0)
            modes_out[mode] = {
                "mean_raw_return": mean_raw.tolist(),
                "mean_scaled_discounted_return": mean_scaled.tolist(),
                "vertex_utility": float(np.dot(pref, mean_scaled)),
            }
    finally:
        env.close()
    return {"vertex_index": vertex_idx, "preference": pref.tolist(), "dev_eval_steps": eval_steps, "modes": modes_out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/protocols/stage1_simplex_face_exposure_v1.json")
    parser.add_argument("--dataset", default=os.environ.get("HEI_DATASET", ""))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("HEI_SEED", "0")))
    parser.add_argument(
        "--device", default=os.environ.get("HEI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    )
    parser.add_argument("--out-dir", default=os.environ.get("HEI_OUT", ""))
    parser.add_argument("--run-id", default=os.environ.get("HEI_RUN_ID", "local-stage1a"))
    parser.add_argument("--compatibility", default=os.environ.get("HEI_COMPATIBILITY", "stage1a-simplex-ppo-v1"))
    parser.add_argument("--override-interactions", type=int, default=0)
    args = parser.parse_args()

    protocol_path = REPO_ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        not protocol.get("execution_authorized")
        or protocol.get("week1_gate_prerequisites", {}).get("gate1b_verdict") != "PASS"
    ):
        raise RuntimeError("Protocol execution_authorized is not True or Gate-1B PASS missing; refusing to run.")

    env_id, kind, spec = parse_dataset_spec(args.dataset)
    benchmarks = {b["env_id"]: b for b in protocol["benchmarks"]}
    if env_id not in benchmarks or env_id not in {"minecart-v0", "mo-lunar-lander-v3"}:
        raise ValueError(f"Unsupported Stage-1A GPU env_id: {env_id}")
    bench = benchmarks[env_id]

    reward_dim = int(bench["reward_dim"])
    state_dim = 7 if env_id == "minecart-v0" else 8
    action_dim = 6 if env_id == "minecart-v0" else 4
    scale = np.asarray(bench["reward_scale"], dtype=np.float32)
    gamma = float(bench["gamma"])
    streams = int(bench["streams"])
    fragment_steps = int(bench["fragment_steps"])
    width = int(bench["width"])
    gae_lambda = float(bench["gae_lambda"])
    total_steps = args.override_interactions if args.override_interactions > 0 else int(bench["training_interactions"])

    vertex_idx: int | None = None
    arm_id: str | None = None
    if kind == "anchor":
        if not spec.startswith("vertex="):
            raise ValueError(f"Anchor spec must be 'vertex=<idx>', got {spec!r}")
        vertex_idx = int(spec.split("=", 1)[1])
        if not 0 <= vertex_idx < reward_dim:
            raise ValueError(f"vertex index {vertex_idx} out of range for d={reward_dim}")
        if args.seed not in bench["anchor_spec"]["anchor_seeds"] and args.override_interactions == 0:
            raise ValueError(f"Anchor seed {args.seed} not in frozen protocol anchor_seeds")
    else:
        arm_id = spec
        if arm_id not in bench["arms_included"]:
            raise ValueError(f"Arm {arm_id!r} not in frozen {env_id} arms_included={bench['arms_included']}")
        if args.seed not in bench["paired_seeds"] and args.override_interactions == 0:
            raise ValueError(f"Training seed {args.seed} not in frozen protocol paired_seeds")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (REPO_ROOT / "artifacts" / "stage1a-gpu-runs" / f"{env_id}_{kind}_{spec}_seed{args.seed}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mirror checkpoints cleanly inside repo's git-ignored artifacts/stage1a-gpu/ for easy auditing & hashing.
    mirror_dir = REPO_ROOT / "artifacts" / "stage1a-gpu" / ("anchors" if kind == "anchor" else "arms") / env_id
    mirror_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{env_id}_{kind}_{spec.replace('=', '')}_seed{args.seed}"

    plan = {
        "env_id": env_id,
        "kind": kind,
        "spec": spec,
        "seed": args.seed,
        "reward_dim": reward_dim,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "training_interactions": total_steps,
        "streams": streams,
        "fragment_steps": fragment_steps,
        "width": width,
        "epochs": 4,
        "minibatches": 8,
        "critic_steps_per_actor": 2,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "adam_eps": 1e-5,
        "clip": 0.2,
        "max_grad_norm": 0.5,
        "popart_rate": 0.001,
        "popart_std_floor": 1e-4,
        "entropy_start": math.log(action_dim),
        "entropy_end": 0.4 if env_id == "minecart-v0" else 0.2,
        "entropy_initial_multiplier": 0.01,
        "entropy_damping": 0.01,
        "entropy_multiplier_lr": 3e-5,
        "hindsight_relabeling": False,
    }

    _seed_all(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model = ActorCritic(state_dim=state_dim, action_dim=action_dim, reward_dim=reward_dim, width=width).to(device)
    init_sha = _model_sha256(model)
    learner = SimplexPPO(model, plan)

    sampler: FaceBalancedPreferenceSampler | None = None
    fixed_pref: np.ndarray | None = None
    if kind == "anchor":
        assert vertex_idx is not None
        fixed_pref = np.zeros(reward_dim, dtype=np.float32)
        fixed_pref[vertex_idx] = 1.0
    else:
        assert arm_id is not None
        sampler = FaceBalancedPreferenceSampler(
            reward_dim=reward_dim,
            mode=arm_id,
            seed=args.seed + 10_000,
            interior_prob=0.50,
            vertex_prob=0.25,
            dirichlet_alpha=0.25,
        )

    envs = [make_mo_env(env_id, seed=args.seed + i) for i in range(streams)]
    observations = np.stack([env.reset(seed=args.seed + i)[0] for i, env in enumerate(envs)])
    if fixed_pref is not None:
        active_prefs = np.tile(fixed_pref, (streams, 1))
    else:
        assert sampler is not None
        active_prefs = np.stack([sampler.sample() for _ in range(streams)])

    step = 0
    rounds = 0
    episodes_completed = 0
    last_loss: dict[str, float] = {}
    started = time.monotonic()

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

                next_obs = np.zeros_like(observations)
                rewards = np.zeros((streams, reward_dim), dtype=np.float32)
                terms = np.zeros(streams, dtype=bool)
                truncs = np.zeros(streams, dtype=bool)
                acts_np = actions.cpu().numpy()
                for i, env in enumerate(envs):
                    nxt, r, tm, tr, _ = env.step(int(acts_np[i]))
                    next_obs[i] = nxt
                    rewards[i] = np.asarray(r, dtype=np.float32) * scale
                    terms[i] = bool(tm)
                    truncs[i] = bool(tr)

                with torch.no_grad():
                    next_values = model.value(torch.tensor(next_obs, dtype=torch.float32, device=device), w_t)

                batch["states"].append(s_t)
                batch["preferences"].append(w_t)
                batch["actions"].append(actions)
                batch["logp"].append(logp)
                batch["values"].append(values)
                batch["next_values"].append(next_values)
                batch["rewards"].append(torch.tensor(rewards, dtype=torch.float32, device=device))
                batch["terminated"].append(torch.tensor(terms, dtype=torch.bool, device=device))
                batch["truncated"].append(torch.tensor(truncs, dtype=torch.bool, device=device))

                observations = next_obs
                step += streams
                for i, env in enumerate(envs):
                    if terms[i] or truncs[i]:
                        episodes_completed += 1
                        observations[i], _ = env.reset()
                        if sampler is not None:
                            active_prefs[i] = sampler.sample()

            last_loss = learner.update({k: torch.stack(v) for k, v in batch.items()}, step)
            rounds += 1
            if step // 250_000 != (step - ticks * streams) // 250_000 or step >= total_steps:
                sps = int(step / max(0.01, time.monotonic() - started))
                print(
                    f"[{tag}] step={step}/{total_steps} sps={sps} actor_upd={learner.actor_updates} "
                    f"entropy={last_loss['entropy']:.4f} eps={episodes_completed}",
                    flush=True,
                )
    finally:
        for env in envs:
            env.close()

    train_wall = time.monotonic() - started
    final_model_sha = _model_sha256(model)
    hardware = torch.cuda.get_device_name(0) if device == "cuda" else platform.machine()
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": device,
        "hardware": hardware,
    }

    ckpt_payload = {
        "model": model.state_dict(),
        "plan": plan,
        "protocol_sha256": _file_sha256(protocol_path),
        "initial_model_sha256": init_sha,
        "final_model_sha256": final_model_sha,
        "actor_updates": learner.actor_updates,
        "critic_updates": learner.critic_updates,
        "training_interactions": step,
        "rounds": rounds,
        "episodes_completed": episodes_completed,
        "preference_exposure": sampler.exposure_summary()
        if sampler is not None
        else {"mode": "fixed_vertex", "vertex": vertex_idx},
        "runtime": runtime,
        "hei_run_id": args.run_id,
    }
    ckpt_out = out_dir / "checkpoint_final.pt"
    tmp_ckpt = out_dir / "checkpoint_final.tmp"
    torch.save(ckpt_payload, tmp_ckpt)
    tmp_ckpt.replace(ckpt_out)
    ckpt_sha = _file_sha256(ckpt_out)

    # Also save an atomic mirror copy in artifacts/stage1a-gpu/ for Stage-1A aggregation.
    mirror_ckpt = mirror_dir / f"{tag}.pt"
    mirror_tmp = mirror_dir / f"{tag}.tmp"
    torch.save(ckpt_payload, mirror_tmp)
    mirror_tmp.replace(mirror_ckpt)

    anchor_eval: dict[str, Any] | None = None
    if kind == "anchor":
        assert vertex_idx is not None
        anchor_eval = evaluate_vertex_anchor(
            model=model,
            env_id=env_id,
            vertex_idx=vertex_idx,
            reward_dim=reward_dim,
            scale=scale.astype(np.float64),
            gamma=gamma,
            device=device,
            eval_seeds=[20269901, 20269902, 20269903, 20269904, 20269905],
        )

    summary = {
        "schema_version": 1,
        "tag": tag,
        "env_id": env_id,
        "kind": kind,
        "spec": spec,
        "seed": args.seed,
        "hei_run_id": args.run_id,
        "protocol_sha256": _file_sha256(protocol_path),
        "initial_model_sha256": init_sha,
        "final_model_sha256": final_model_sha,
        "checkpoint_sha256": ckpt_sha,
        "mirror_checkpoint": str(mirror_ckpt.relative_to(REPO_ROOT)),
        "training_interactions": step,
        "actor_updates": learner.actor_updates,
        "critic_updates": learner.critic_updates,
        "rounds": rounds,
        "episodes_completed": episodes_completed,
        "train_wall_seconds": round(train_wall, 3),
        "gpu_hours": round(train_wall / 3600.0, 5),
        "final_test_steps": 0,
        "final_loss": last_loss,
        "preference_exposure": ckpt_payload["preference_exposure"],
        "anchor_dev_eval": anchor_eval,
        "runtime": runtime,
    }
    summary_path = mirror_dir / f"{tag}_receipt.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    metrics_out: dict[str, float] = {
        "train_wall_seconds": round(train_wall, 3),
        "train_sps": round(step / max(0.01, train_wall), 1),
        "final_entropy": float(last_loss["entropy"]),
        "gpu_hours": round(train_wall / 3600.0, 5),
    }
    if anchor_eval is not None:
        metrics_out["anchor_argmax_vertex_utility"] = float(anchor_eval["modes"]["argmax"]["vertex_utility"])
        metrics_out["anchor_sampled_vertex_utility"] = float(anchor_eval["modes"]["sampled"]["vertex_utility"])

    result = {
        "schema": 1,
        "run_id": args.run_id,
        "dataset": args.dataset,
        "compatibility": args.compatibility,
        "seed": args.seed,
        "epoch": 1,
        "runtime": runtime,
        "resumed_from_epoch": 0,
        "previous_runtime": None,
        "metrics": metrics_out,
        "artifacts": {"checkpoint_final.pt": ckpt_sha},
    }
    tmp_res = out_dir / "result.tmp"
    tmp_res.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    tmp_res.replace(out_dir / "result.json")


if __name__ == "__main__":
    main()

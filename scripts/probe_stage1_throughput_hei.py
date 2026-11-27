#!/usr/bin/env python3
"""Short timed throughput probe for minecart-v0 (d=3) and full-simplex mo-lunar-lander-v3 (d=4) via hei-research.

Engineering throughput witness ONLY (not reportable task evidence, zero final-test access).
Measures live WSL CUDA training throughput + evaluation overhead to verify the <= 2.0 GPU-hours/run ceiling
before freezing the Stage-1 Protocol JSON.
"""

from __future__ import annotations

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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.envs.wrapper import make_mo_env  # noqa: E402
from src.evaluation.protocol import load_protocol, resolve_reward_scale  # noqa: E402
from src.evaluation.simplex_metrics import FaceBalancedPreferenceSampler  # noqa: E402
from src.models.face_exposure_ppo import SimplexPPO  # noqa: E402
from src.models.onpolicy import ActorCritic  # noqa: E402


def probe_env(
    env_id: str,
    protocol_path: str,
    reward_dim: int,
    state_dim: int,
    action_dim: int,
    streams: int,
    fragment_steps: int,
    probe_interactions: int,
    target_interactions: int,
    eval_episodes_per_checkpoint: int,
    eval_checkpoints: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    protocol, _ = load_protocol(REPO_ROOT / protocol_path, "validation")
    scale = resolve_reward_scale(protocol)
    plan = {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "training_interactions": target_interactions,
        "streams": streams,
        "fragment_steps": fragment_steps,
        "width": 256,
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
        "entropy_end": 0.2 if action_dim == 4 else 0.4,
        "entropy_initial_multiplier": 0.01,
        "entropy_damping": 0.01,
        "entropy_multiplier_lr": 3e-5,
    }
    model = ActorCritic(state_dim=state_dim, action_dim=action_dim, reward_dim=reward_dim, width=256).to(device)
    learner = SimplexPPO(model, plan)
    sampler = FaceBalancedPreferenceSampler(reward_dim=reward_dim, mode="fixed_face_balanced", seed=seed + 100)

    envs = [make_mo_env(env_id, seed=seed + i) for i in range(streams)]
    observations = np.stack([env.reset(seed=seed + i)[0] for i, env in enumerate(envs)])
    active_prefs = np.stack([sampler.sample() for _ in range(streams)])

    step = 0
    rounds = 0
    if device == "cuda":
        torch.cuda.synchronize()
    train_start = time.monotonic()
    try:
        while step < probe_interactions:
            ticks = min(fragment_steps, (probe_interactions - step) // streams)
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
                        observations[i], _ = env.reset()
                        active_prefs[i] = sampler.sample()

            learner.update({k: torch.stack(v) for k, v in batch.items()}, step)
            rounds += 1
        if device == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.monotonic() - train_start

        # Short evaluation throughput probe (20 rollouts capped at 250 steps each for timing)
        eval_env = make_mo_env(env_id, seed=seed + 500)
        eval_steps = 0
        eval_start = time.monotonic()
        model.eval()
        try:
            with torch.no_grad():
                for ep in range(20):
                    obs, _ = eval_env.reset(seed=seed + 500 + ep)
                    w_t = torch.tensor(sampler.sample(), dtype=torch.float32, device=device).unsqueeze(0)
                    for _ in range(250):
                        s_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        act = int(model.distribution(s_t, w_t).logits.argmax().item())
                        obs, _, tm, tr, _ = eval_env.step(act)
                        eval_steps += 1
                        if tm or tr:
                            break
        finally:
            eval_env.close()
        if device == "cuda":
            torch.cuda.synchronize()
        eval_seconds = time.monotonic() - eval_start
    finally:
        for env in envs:
            env.close()

    train_sps = step / max(1e-6, train_seconds)
    eval_sps = eval_steps / max(1e-6, eval_seconds)
    projected_train_hours = (target_interactions / train_sps) / 3600.0
    # Conservative 250 avg eval steps/episode across final-only + dev checks
    projected_eval_steps = eval_episodes_per_checkpoint * eval_checkpoints * 250
    projected_eval_hours = (projected_eval_steps / eval_sps) / 3600.0
    projected_total_hours = projected_train_hours + projected_eval_hours

    return {
        "env_id": env_id,
        "reward_dim": reward_dim,
        "probe_interactions": step,
        "probe_rounds": rounds,
        "probe_train_seconds": round(train_seconds, 3),
        "train_sps": round(train_sps, 1),
        "probe_eval_steps": eval_steps,
        "probe_eval_seconds": round(eval_seconds, 3),
        "eval_sps": round(eval_sps, 1),
        "target_interactions": target_interactions,
        "projected_train_gpu_hours": round(projected_train_hours, 4),
        "projected_eval_gpu_hours": round(projected_eval_hours, 4),
        "projected_total_gpu_hours": round(projected_total_hours, 4),
        "under_2_gpu_hours_ceiling": bool(projected_total_hours <= 2.0),
    }


def main() -> None:
    out_dir = Path(os.environ.get("HEI_OUT", str(REPO_ROOT / "artifacts" / "stage1-throughput-probe-v1")))
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(os.environ.get("HEI_SEED", "20269901"))
    device = os.environ.get("HEI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    run_id = os.environ.get("HEI_RUN_ID", "local-manual-probe")
    dataset = os.environ.get("HEI_DATASET", "minecart-v0_and_mo-lunar-lander-v3_full-simplex")
    compatibility = os.environ.get("HEI_COMPATIBILITY", "simplex-ppo-throughput-v1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    hardware = torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else platform.machine()
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": device,
        "hardware": hardware,
    }

    minecart_4m = probe_env(
        env_id="minecart-v0",
        protocol_path="configs/protocols/course_minecart_v0.json",
        reward_dim=3,
        state_dim=7,
        action_dim=6,
        streams=8,
        fragment_steps=64,
        probe_interactions=16384,
        target_interactions=4_000_000,
        eval_episodes_per_checkpoint=426,
        eval_checkpoints=1,
        seed=seed,
        device=device,
    )
    minecart_2m_hours = round(
        minecart_4m["projected_train_gpu_hours"] * 0.5 + minecart_4m["projected_eval_gpu_hours"], 4
    )
    minecart_4m["projected_2m_total_gpu_hours"] = minecart_2m_hours

    lunar_4m = probe_env(
        env_id="mo-lunar-lander-v3",
        protocol_path="configs/protocols/course_mo_lunar_lander_v3.json",
        reward_dim=4,
        state_dim=8,
        action_dim=4,
        streams=8,
        fragment_steps=128,
        probe_interactions=16384,
        target_interactions=4_000_000,
        eval_episodes_per_checkpoint=660,
        eval_checkpoints=1,
        seed=seed + 1000,
        device=device,
    )

    detail_payload = {
        "schema_version": 1,
        "kind": "stage1_throughput_probe",
        "engineering_only": True,
        "final_test_steps": 0,
        "run_id": run_id,
        "runtime": runtime,
        "probes": {
            "minecart-v0": minecart_4m,
            "mo-lunar-lander-v3": lunar_4m,
        },
    }
    detail_path = out_dir / "throughput_probe.json"
    detail_bytes = (json.dumps(detail_payload, indent=2) + "\n").encode("utf-8")
    detail_path.write_bytes(detail_bytes)
    detail_sha = hashlib.sha256(detail_bytes).hexdigest()

    # Schema-1 hei-research result contract
    result = {
        "schema": 1,
        "run_id": run_id,
        "dataset": dataset,
        "compatibility": compatibility,
        "seed": seed,
        "epoch": 1,
        "runtime": runtime,
        "resumed_from_epoch": 0,
        "previous_runtime": None,
        "metrics": {
            "minecart_train_sps": minecart_4m["train_sps"],
            "minecart_projected_4m_gpu_hours": minecart_4m["projected_total_gpu_hours"],
            "minecart_projected_2m_gpu_hours": minecart_2m_hours,
            "lunar_train_sps": lunar_4m["train_sps"],
            "lunar_projected_4m_gpu_hours": lunar_4m["projected_total_gpu_hours"],
        },
        "artifacts": {"throughput_probe.json": detail_sha},
    }
    tmp = out_dir / "result.tmp"
    tmp.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out_dir / "result.json")
    print(json.dumps(detail_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

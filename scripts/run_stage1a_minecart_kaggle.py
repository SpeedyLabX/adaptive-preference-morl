#!/usr/bin/env python3
"""Self-contained Stage-1A Minecart GPU runner for offline Kaggle T4 / Colab / Local via hei-research.

Requires ONLY Python stdlib + numpy + scipy + torch (zero external repo imports or internet access),
so hei-research can execute it cleanly on Kaggle T4 kernels (enable_internet=False).

Bit-exact equivalence with mo-gymnasium==1.3.2 minecart-v0 and src/models/{onpolicy,face_exposure_ppo}.py
is verified by tests/test_stage1a_gpu_pipeline.py. Strictly maintains final_test_steps = 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats
import torch
from torch import nn
from torch.distributions import Categorical

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Frozen mo-gymnasium==1.3.2 mine_config.json
# SHA-256: 9cbaea02440cb21bd799ede48b199799a655154526e9e7daf54225f5463037d9
MINECART_CONFIG: dict[str, Any] = {
    "capacity": 1.5,
    "mine_cnt": 5,
    "ore_cnt": 2,
    "mines": [
        {"x": 0.16, "y": 0.84, "distributions": [[0.2, 0.05], [0.0, 0.05]]},
        {"x": 0.5, "y": 0.84, "distributions": [[0.15, 0.05], [0.1, 0.05]]},
        {"x": 0.84, "y": 0.84, "distributions": [[0.2, 0.05], [0.2, 0.05]]},
        {"x": 0.84, "y": 0.5, "distributions": [[0.1, 0.05], [0.15, 0.05]]},
        {"x": 0.84, "y": 0.16, "distributions": [[0.0, 0.05], [0.2, 0.05]]},
    ],
}

# Frozen Stage-1A Minecart protocol parameters (configs/protocols/stage1_simplex_face_exposure_v1.json).
MINECART_PROTOCOL: dict[str, Any] = {
    "protocol_id": "stage1-simplex-face-exposure-v1",
    "env_id": "minecart-v0",
    "reward_dim": 3,
    "state_dim": 7,
    "action_dim": 6,
    "gamma": 0.99,
    "reward_scale": [0.9659120784281335, 0.9659120784281335, 0.8247459754193536],
    "training_interactions": 4_000_000,
    "streams": 8,
    "fragment_steps": 64,
    "width": 256,
    "gae_lambda": 1.0,
    "anchor_seeds": [20269281, 20269282, 20269283],
    "paired_seeds": [
        20269201,
        20269202,
        20269203,
        20269204,
        20269205,
        20269206,
        20269207,
        20269208,
        20269209,
        20269210,
    ],
    "arms_included": ["uniform", "dirichlet_sparse", "fixed_face_balanced"],
}

EPS_SPEED = 0.001
HOME_POS = (0.0, 0.0)
ROTATION = 10
MAX_SPEED = 1.0
FUEL_MINE = -0.05
FUEL_ACC = -0.025
FUEL_IDLE = -0.005
ACT_MINE = 0
ACT_LEFT = 1
ACT_RIGHT = 2
ACT_ACCEL = 3
ACT_BRAKE = 4
ACT_NONE = 5
MINE_RADIUS = 0.14
BASE_RADIUS = 0.15
ACCELERATION = 0.0075
DECELERATION = 1.0


def _mag(vec: np.ndarray) -> float:
    return float(np.sqrt(np.dot(vec, vec)))


def _clip(val: float, lo: float, hi: float) -> float:
    return lo if val <= lo else hi if val >= hi else val


class _StandaloneMine:
    def __init__(self, x: float, y: float, dists: list[list[float]]) -> None:
        self.pos = np.array([x, y], dtype=np.float64)
        self.distributions = [scipy.stats.norm(d[0], d[1]) for d in dists]

    def distance(self, cart_pos: np.ndarray) -> float:
        return _mag(cart_pos - self.pos)

    def mineable(self, cart_pos: np.ndarray) -> bool:
        return self.distance(cart_pos) <= MINE_RADIUS

    def mine(self) -> list[float]:
        return [max(0.0, float(dist.rvs())) for dist in self.distributions]


class _StandaloneCart:
    def __init__(self, ore_cnt: int) -> None:
        self.ore_cnt = ore_cnt
        self.pos = np.array([0.0, 0.0], dtype=np.float64)
        self.speed = 0.0
        self.angle = 45.0
        self.content = np.zeros(ore_cnt, dtype=np.float64)
        self.departed = False

    def accelerate(self, accel: float) -> None:
        self.speed = _clip(self.speed + accel, 0.0, MAX_SPEED)

    def rotate(self, rot: float) -> None:
        self.angle = (self.angle + rot) % 360.0

    def step(self) -> bool:
        pre = np.copy(self.pos)
        if self.speed < EPS_SPEED:
            return False
        x_vel = self.speed * math.cos(self.angle * math.pi / 180.0)
        y_vel = self.speed * math.sin(self.angle * math.pi / 180.0)
        x, y = float(self.pos[0]), float(self.pos[1])
        if y != 0.0 and y != 1.0 and (y_vel > EPS_SPEED or y_vel < -EPS_SPEED):
            if x == 1.0 and x_vel > 0.0:
                self.angle += math.copysign(ROTATION, y_vel)
            if x == 0.0 and x_vel < 0.0:
                self.angle -= math.copysign(ROTATION, y_vel)
        if x != 0.0 and x != 1.0 and (x_vel > EPS_SPEED or x_vel < -EPS_SPEED):
            if y == 1.0 and y_vel > 0.0:
                self.angle -= math.copysign(ROTATION, x_vel)
            if y == 0.0 and y_vel < 0.0:
                self.angle += math.copysign(ROTATION, x_vel)
        self.pos[0] = _clip(x + x_vel, 0.0, 1.0)
        self.pos[1] = _clip(y + y_vel, 0.0, 1.0)
        self.speed = _mag(pre - self.pos)
        return True


class StandaloneMinecart:
    """Pure NumPy/SciPy replica of mo-gymnasium==1.3.2 minecart-v0 (max_episode_steps=1000)."""

    def __init__(self, frame_skip: int = 4, max_episode_steps: int = 1000) -> None:
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.ore_cnt = int(MINECART_CONFIG["ore_cnt"])
        self.capacity = float(MINECART_CONFIG["capacity"])
        self.mines = [
            _StandaloneMine(float(m["x"]), float(m["y"]), m["distributions"]) for m in MINECART_CONFIG["mines"]
        ]
        self.cart = _StandaloneCart(self.ore_cnt)
        self.end = False
        self.elapsed_steps = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        _ = seed
        self.cart.content = np.zeros(self.ore_cnt, dtype=np.float64)
        self.cart.pos = np.array(HOME_POS, dtype=np.float64)
        self.cart.speed = 0.0
        self.cart.angle = 45.0
        self.cart.departed = False
        self.end = False
        self.elapsed_steps = 0
        return self._get_state(), {}

    def _get_state(self) -> np.ndarray:
        rad = math.radians(self.cart.angle)
        ang = np.array([math.sin(rad), math.cos(rad)], dtype=np.float32)
        return np.concatenate(
            (self.cart.pos, np.array([self.cart.speed]), ang, self.cart.content / self.capacity),
            dtype=np.float32,
        )

    def _mine(self) -> bool:
        if self.cart.speed < EPS_SPEED:
            mine = min(self.mines, key=lambda m: m.distance(self.cart.pos))
            if mine.mineable(self.cart.pos):
                cart_free = self.capacity - float(np.sum(self.cart.content))
                mined = np.asarray(mine.mine(), dtype=np.float64)
                total_mined = float(np.sum(mined))
                if total_mined > cart_free and total_mined > 0.0:
                    mined = mined * (cart_free / total_mined)
                self.cart.content += mined
                if float(np.sum(mined)) > 0.0:
                    return True
        return False

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        reward = np.zeros(self.ore_cnt + 1, dtype=np.float32)
        reward[-1] = FUEL_IDLE * self.frame_skip
        if action == ACT_ACCEL:
            reward[-1] += FUEL_ACC * self.frame_skip
        elif action == ACT_MINE:
            reward[-1] += FUEL_MINE * self.frame_skip

        for _ in range(self.frame_skip):
            if action == ACT_LEFT:
                self.cart.rotate(-ROTATION)
            elif action == ACT_RIGHT:
                self.cart.rotate(ROTATION)
            elif action == ACT_ACCEL:
                self.cart.accelerate(ACCELERATION)
            elif action == ACT_BRAKE:
                self.cart.accelerate(-DECELERATION)
            elif action == ACT_MINE:
                self._mine()

            if self.end:
                break

            self.cart.step()
            dist_home = _mag(self.cart.pos - np.asarray(HOME_POS, dtype=np.float64))
            if dist_home < BASE_RADIUS:
                if self.cart.departed:
                    self.end = True
                    reward[: self.ore_cnt] += self.cart.content.astype(np.float32)
                    self.cart.content = np.zeros(self.ore_cnt, dtype=np.float64)
            else:
                self.cart.departed = True

        self.elapsed_steps += 1
        truncated = bool(self.elapsed_steps >= self.max_episode_steps)
        return self._get_state(), reward, bool(self.end), truncated, {}

    def close(self) -> None:
        return None


class MultiBody(nn.Module):
    def __init__(self, state_dim: int, reward_dim: int, output_dim: int, width: int) -> None:
        super().__init__()
        self.bodies = nn.ModuleList([nn.Linear(state_dim, width) for _ in range(reward_dim)])
        self.hidden = nn.Linear(width, width)
        self.head = nn.Linear(width, output_dim)
        for layer in [*self.bodies, self.hidden, self.head]:
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor, preference: torch.Tensor) -> torch.Tensor:
        bodies = torch.stack([torch.relu(body(state)) for body in self.bodies], dim=-2)
        merged = (bodies * preference.unsqueeze(-1)).sum(-2)
        return self.head(torch.relu(self.hidden(merged)))


class ActorCritic(nn.Module):
    mean: torch.Tensor
    second: torch.Tensor
    std: torch.Tensor

    def __init__(self, state_dim: int = 7, action_dim: int = 6, reward_dim: int = 3, width: int = 256) -> None:
        super().__init__()
        self.actor = MultiBody(state_dim, reward_dim, action_dim, width)
        self.critic = MultiBody(state_dim, reward_dim, reward_dim, width)
        nn.init.orthogonal_(self.actor.head.weight, 0.01)
        nn.init.orthogonal_(self.critic.head.weight, 1.0)
        self.register_buffer("mean", torch.zeros(reward_dim))
        self.register_buffer("second", torch.ones(reward_dim))
        self.register_buffer("std", torch.ones(reward_dim))

    def distribution(self, state: torch.Tensor, preference: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(state, preference))

    def value(self, state: torch.Tensor, preference: torch.Tensor) -> torch.Tensor:
        return self.critic(state, preference) * self.std + self.mean

    @torch.no_grad()
    def update_popart(self, targets: torch.Tensor, rate: float, floor: float) -> None:
        old_mean, old_std = self.mean.clone(), self.std.clone()
        self.mean.lerp_(targets.mean(0), rate)
        self.second.lerp_(targets.square().mean(0), rate)
        self.std.copy_((self.second - self.mean.square()).clamp_min(floor**2).sqrt())
        self.critic.head.weight.mul_((old_std / self.std).unsqueeze(1))
        self.critic.head.bias.copy_((old_std * self.critic.head.bias + old_mean - self.mean) / self.std)


def _gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    result = torch.empty_like(rewards)
    carry = torch.zeros_like(rewards[-1])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (~terminated[t]).unsqueeze(-1) * next_values[t] - values[t]
        carry = delta + gamma * lam * (~(terminated[t] | truncated[t])).unsqueeze(-1) * carry
        result[t] = carry
    return result


class SimplexPPO:
    def __init__(self, model: ActorCritic, plan: dict[str, Any]) -> None:
        self.model = model
        self.plan = plan
        self.actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=plan["actor_lr"], eps=plan["adam_eps"])
        self.critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=plan["critic_lr"], eps=plan["adam_eps"])
        self.multiplier = float(plan["entropy_initial_multiplier"])
        self.actor_updates = 0
        self.critic_updates = 0
        self.action_dim = int(model.actor.head.out_features)
        self.gae_lambda = float(plan.get("gae_lambda", 1.0))

    def update(self, batch: dict[str, torch.Tensor], step: int) -> dict[str, float]:
        p, model = self.plan, self.model
        advantages = _gae_advantages(
            batch["rewards"],
            batch["values"],
            batch["next_values"],
            batch["terminated"],
            batch["truncated"],
            float(p["gamma"]),
            self.gae_lambda,
        )
        targets = (advantages + batch["values"]).flatten(0, 1)
        flat = {k: v.flatten(0, 1) for k, v in batch.items()}
        advantage = ((targets - flat["values"]) * flat["preferences"]).sum(-1)
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)

        model.update_popart(targets, float(p["popart_rate"]), float(p["popart_std_floor"]))
        normalized_targets = (targets - model.mean) / model.std

        entropy_start = float(p.get("entropy_start", math.log(self.action_dim)))
        entropy_end = float(p["entropy_end"])
        target_entropy = entropy_start + min(step / float(p["training_interactions"]), 1.0) * (
            entropy_end - entropy_start
        )

        diagnostics = []
        for _ in range(int(p["epochs"])):
            order = torch.randperm(len(targets), device=targets.device)
            for idx in order.tensor_split(int(p["minibatches"])):
                s, w = flat["states"][idx], flat["preferences"][idx]
                for _ in range(int(p["critic_steps_per_actor"])):
                    loss_c = (model.critic(s, w) - normalized_targets[idx]).square().mean()
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    loss_c.backward()
                    nn.utils.clip_grad_norm_(
                        model.critic.parameters(), float(p["max_grad_norm"]), error_if_nonfinite=True
                    )
                    self.critic_optimizer.step()
                    self.critic_updates += 1

                dist = model.distribution(s, w)
                logp = dist.log_prob(flat["actions"][idx])
                ratio = (logp - flat["logp"][idx]).exp()
                adv_i = advantage[idx]
                clip_eps = float(p["clip"])
                loss_pi = -torch.minimum(ratio * adv_i, ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv_i).mean()
                entropy = dist.entropy().mean()
                error = target_entropy - float(entropy.detach())
                coeff = max(0.0, self.multiplier + float(p["entropy_damping"]) * error)
                loss = loss_pi - coeff * entropy
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.actor.parameters(), float(p["max_grad_norm"]), error_if_nonfinite=True)
                self.actor_optimizer.step()
                self.multiplier = max(0.0, self.multiplier + float(p["entropy_multiplier_lr"]) * error)
                self.actor_updates += 1
                diagnostics.append(torch.stack([loss_pi.detach(), loss_c.detach(), entropy.detach()]))

        means = torch.stack(diagnostics).mean(0).cpu().tolist()
        if not all(math.isfinite(x) for x in [*means, self.multiplier]):
            raise FloatingPointError("Nonfinite SimplexPPO diagnostics")
        return {
            "policy_loss": float(means[0]),
            "value_loss": float(means[1]),
            "entropy": float(means[2]),
            "entropy_target": target_entropy,
            "entropy_multiplier": self.multiplier,
        }


class StaticSampler:
    def __init__(self, reward_dim: int, mode: str, seed: int, dirichlet_alpha: float = 0.25) -> None:
        self.reward_dim = reward_dim
        self.mode = mode
        self.dirichlet_alpha = dirichlet_alpha
        self.rng = np.random.default_rng(seed)
        self.stratum_counts = {"interior": 0, "dirichlet_sparse": 0, "vertex": 0, "sparse_face": 0}
        self.vertex_counts = np.zeros(reward_dim, dtype=np.int64)

    def sample(self) -> np.ndarray:
        if self.mode == "uniform":
            self.stratum_counts["interior"] += 1
            return self.rng.dirichlet(np.ones(self.reward_dim, dtype=np.float64)).astype(np.float32)
        if self.mode == "dirichlet_sparse":
            self.stratum_counts["dirichlet_sparse"] += 1
            return self.rng.dirichlet(np.full(self.reward_dim, self.dirichlet_alpha, dtype=np.float64)).astype(
                np.float32
            )
        draw = float(self.rng.random())
        if draw < 0.50:
            self.stratum_counts["interior"] += 1
            return self.rng.dirichlet(np.ones(self.reward_dim, dtype=np.float64)).astype(np.float32)
        if draw < 0.75:
            v = int(self.rng.integers(self.reward_dim))
            w = np.zeros(self.reward_dim, dtype=np.float32)
            w[v] = 1.0
            self.stratum_counts["vertex"] += 1
            self.vertex_counts[v] += 1
            return w
        k = int(self.rng.integers(2, self.reward_dim))
        subset = self.rng.choice(self.reward_dim, size=k, replace=False)
        w64 = np.zeros(self.reward_dim, dtype=np.float64)
        w64[subset] = self.rng.dirichlet(np.ones(k, dtype=np.float64))
        w64 /= w64.sum()
        self.stratum_counts["sparse_face"] += 1
        return w64.astype(np.float32)

    def exposure_summary(self) -> dict[str, Any]:
        total = sum(self.stratum_counts.values())
        return {
            "mode": self.mode,
            "dirichlet_alpha": self.dirichlet_alpha if self.mode == "dirichlet_sparse" else None,
            "total_draws": total,
            "stratum_counts": dict(self.stratum_counts),
            "stratum_fractions": {k: (v / total if total else 0.0) for k, v in self.stratum_counts.items()},
            "vertex_counts": self.vertex_counts.tolist(),
        }


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


@torch.no_grad()
def _evaluate_vertex_anchor(
    model: ActorCritic,
    vertex_idx: int,
    scale: np.ndarray,
    gamma: float,
    device: str,
    eval_seeds: list[int],
) -> dict[str, Any]:
    env = StandaloneMinecart()
    pref = np.zeros(3, dtype=np.float32)
    pref[vertex_idx] = 1.0
    w_t = torch.tensor(pref, dtype=torch.float32, device=device).unsqueeze(0)
    model.eval()
    modes_out: dict[str, Any] = {}
    eval_steps = 0
    for mode in ("argmax", "sampled"):
        raw_list: list[np.ndarray] = []
        scaled_list: list[np.ndarray] = []
        for rep_idx, s in enumerate(eval_seeds):
            torch.manual_seed(s + vertex_idx * 100 + rep_idx)
            np.random.seed(s + vertex_idx * 100 + rep_idx)
            obs, _ = env.reset(seed=s)
            raw = np.zeros(3, dtype=np.float64)
            disc = np.zeros(3, dtype=np.float64)
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
    return {"vertex_index": vertex_idx, "preference": pref.tolist(), "dev_eval_steps": eval_steps, "modes": modes_out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=os.environ.get("HEI_DATASET", ""))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("HEI_SEED", "0")))
    parser.add_argument(
        "--device", default=os.environ.get("HEI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    )
    parser.add_argument("--out-dir", default=os.environ.get("HEI_OUT", ""))
    parser.add_argument("--run-id", default=os.environ.get("HEI_RUN_ID", "kaggle-stage1a"))
    parser.add_argument("--compatibility", default=os.environ.get("HEI_COMPATIBILITY", "stage1a-simplex-ppo-v1"))
    parser.add_argument("--override-interactions", type=int, default=0)
    args = parser.parse_args()

    parts = args.dataset.strip().split("|")
    if len(parts) != 3 or parts[0] != "minecart-v0" or parts[1] not in {"anchor", "arm"}:
        raise ValueError(f"Expected 'minecart-v0|<anchor|arm>|<spec>', got {args.dataset!r}")
    env_id, kind, spec = parts

    vertex_idx: int | None = None
    arm_id: str | None = None
    if kind == "anchor":
        if not spec.startswith("vertex="):
            raise ValueError(f"Expected vertex=<0|1|2>, got {spec!r}")
        vertex_idx = int(spec.split("=", 1)[1])
        if vertex_idx not in {0, 1, 2}:
            raise ValueError(f"Invalid Minecart vertex: {vertex_idx}")
        if args.seed not in MINECART_PROTOCOL["anchor_seeds"] and args.override_interactions == 0:
            raise ValueError(f"Anchor seed {args.seed} not in frozen Minecart anchor_seeds")
    else:
        arm_id = spec
        if arm_id not in MINECART_PROTOCOL["arms_included"]:
            raise ValueError(f"Arm {arm_id!r} not in frozen Minecart arms")
        if args.seed not in MINECART_PROTOCOL["paired_seeds"] and args.override_interactions == 0:
            raise ValueError(f"Training seed {args.seed} not in frozen Minecart paired_seeds")

    out_dir = Path(args.out_dir) if args.out_dir else Path("./hei_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{env_id}_{kind}_{spec.replace('=', '')}_seed{args.seed}"

    total_steps = (
        args.override_interactions
        if args.override_interactions > 0
        else int(MINECART_PROTOCOL["training_interactions"])
    )
    streams = int(MINECART_PROTOCOL["streams"])
    fragment_steps = int(MINECART_PROTOCOL["fragment_steps"])
    width = int(MINECART_PROTOCOL["width"])
    gamma = float(MINECART_PROTOCOL["gamma"])
    scale = np.asarray(MINECART_PROTOCOL["reward_scale"], dtype=np.float32)

    plan = {
        "env_id": env_id,
        "kind": kind,
        "spec": spec,
        "seed": args.seed,
        "reward_dim": 3,
        "gamma": gamma,
        "gae_lambda": float(MINECART_PROTOCOL["gae_lambda"]),
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
        "entropy_start": math.log(6),
        "entropy_end": 0.4,
        "entropy_initial_multiplier": 0.01,
        "entropy_damping": 0.01,
        "entropy_multiplier_lr": 3e-5,
        "hindsight_relabeling": False,
    }

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model = ActorCritic(state_dim=7, action_dim=6, reward_dim=3, width=width).to(device)
    init_sha = _model_sha256(model)
    learner = SimplexPPO(model, plan)

    sampler: StaticSampler | None = None
    fixed_pref: np.ndarray | None = None
    if kind == "anchor":
        assert vertex_idx is not None
        fixed_pref = np.zeros(3, dtype=np.float32)
        fixed_pref[vertex_idx] = 1.0
    else:
        assert arm_id is not None
        sampler = StaticSampler(reward_dim=3, mode=arm_id, seed=args.seed + 10_000, dirichlet_alpha=0.25)

    envs = [StandaloneMinecart() for _ in range(streams)]
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
            rewards = np.zeros((streams, 3), dtype=np.float32)
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

    anchor_eval: dict[str, Any] | None = None
    if kind == "anchor":
        assert vertex_idx is not None
        anchor_eval = _evaluate_vertex_anchor(
            model=model,
            vertex_idx=vertex_idx,
            scale=scale.astype(np.float64),
            gamma=gamma,
            device=device,
            eval_seeds=[20269901, 20269902, 20269903, 20269904, 20269905],
        )

    receipt = {
        "schema_version": 1,
        "tag": tag,
        "env_id": env_id,
        "kind": kind,
        "spec": spec,
        "seed": args.seed,
        "hei_run_id": args.run_id,
        "standalone_kaggle_runner": True,
        "initial_model_sha256": init_sha,
        "final_model_sha256": final_model_sha,
        "checkpoint_sha256": ckpt_sha,
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
    receipt_out = out_dir / "receipt.json"
    receipt_out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    receipt_sha = _file_sha256(receipt_out)

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
        "artifacts": {
            "checkpoint_final.pt": ckpt_sha,
            "receipt.json": receipt_sha,
        },
    }
    tmp_res = out_dir / "result.tmp"
    tmp_res.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    tmp_res.replace(out_dir / "result.json")


if __name__ == "__main__":
    main()

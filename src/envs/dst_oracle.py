"""Exact discounted oracle and grid one-hot encoder for mo-gymnasium deep-sea-treasure-v0 (d=2)."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from src.envs.wrapper import make_mo_env


def encode_dst_state(obs: np.ndarray, rows: int = 11, cols: int = 11) -> np.ndarray:
    """Encode DeepSeaTreasure [row, col] as a deterministic one-hot vector of size rows * cols."""
    r = int(obs[0])
    c = int(obs[1])
    if not (0 <= r < rows and 0 <= c < cols):
        raise ValueError(f"Invalid DST observation {obs.tolist()} for grid {rows}x{cols}")
    vec = np.zeros(rows * cols, dtype=np.float32)
    vec[r * cols + c] = 1.0
    return vec


def extract_dst_oracle(gamma: float = 0.99) -> dict[str, Any]:
    """Verify exact discounted Pareto front of deep-sea-treasure-v0 via shortest-path BFS rollouts."""
    env = make_mo_env("deep-sea-treasure-v0")
    try:
        core: Any = env.unwrapped
        sea_map = np.asarray(core.sea_map, dtype=np.float64)
        rows, cols = sea_map.shape
        start = (0, 0)

        # BFS shortest paths from (0, 0) to every treasure cell (sea_map[r, c] > 0).
        action_deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        queue: deque[tuple[tuple[int, int], list[int]]] = deque([(start, [])])
        visited: set[tuple[int, int]] = {start}
        treasure_paths: dict[tuple[int, int], list[int]] = {}

        while queue:
            (r, c), path = queue.popleft()
            if sea_map[r, c] > 0.0:
                treasure_paths[(r, c)] = path
                continue  # Episode terminates on treasure.
            for act, (dr, dc) in action_deltas.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and sea_map[nr, nc] >= 0.0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), [*path, act]))

        rollout_returns: list[np.ndarray] = []
        raw_returns: list[np.ndarray] = []
        for cell in sorted(treasure_paths):
            path = treasure_paths[cell]
            obs, _ = env.reset(seed=0)
            raw = np.zeros(2, dtype=np.float64)
            disc = np.zeros(2, dtype=np.float64)
            discount = 1.0
            terminated = False
            for act in path:
                obs, reward, terminated, _, _ = env.step(act)
                rv = np.asarray(reward, dtype=np.float64)
                raw += rv
                disc += discount * rv
                discount *= gamma
            if not terminated or (int(obs[0]), int(obs[1])) != cell:
                raise RuntimeError(f"BFS path did not terminate at DST treasure {cell}; got {obs}")
            raw_returns.append(raw)
            rollout_returns.append(disc)

        disc_arr = np.asarray(rollout_returns, dtype=np.float64)
        raw_arr = np.asarray(raw_returns, dtype=np.float64)
        order = np.argsort(disc_arr[:, 0])
        disc_arr = disc_arr[order]
        raw_arr = raw_arr[order]

        env_pf = np.asarray(core.pareto_front(gamma), dtype=np.float64)
        env_pf = env_pf[np.argsort(env_pf[:, 0])]
        max_abs_diff = float(np.max(np.abs(disc_arr - env_pf)))
        if max_abs_diff > 1e-4:
            raise ValueError(f"DST BFS oracle disagreed with env.pareto_front({gamma}): max_diff={max_abs_diff}")
    finally:
        env.close()

    return {
        "env_id": "deep-sea-treasure-v0",
        "reward_dim": 2,
        "grid_shape": [int(rows), int(cols)],
        "gamma": gamma,
        "num_treasures": len(disc_arr),
        "discounted_pareto_front": disc_arr,
        "raw_pareto_front": raw_arr,
        "verified_max_abs_diff_vs_env_pareto_front": max_abs_diff,
    }

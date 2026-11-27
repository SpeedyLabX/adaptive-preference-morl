"""Exact discounted oracle and node one-hot encoder for mo-gymnasium fruit-tree-v0 (d=6)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.envs.wrapper import make_mo_env


def encode_fruit_tree_node(obs: np.ndarray, tree_depth: int = 6) -> np.ndarray:
    """Encode FruitTreeEnv [row, pos_in_row] as a deterministic one-hot vector over 2^(depth+1) - 1 nodes."""
    row = int(obs[0])
    pos = int(obs[1])
    num_nodes = (1 << (tree_depth + 1)) - 1
    node_idx = (1 << row) - 1 + pos
    if not 0 <= node_idx < num_nodes:
        raise ValueError(f"Invalid fruit-tree observation {obs.tolist()} for depth={tree_depth}")
    encoded = np.zeros(num_nodes, dtype=np.float32)
    encoded[node_idx] = 1.0
    return encoded


def extract_fruit_tree_oracle(tree_depth: int = 6, gamma: float = 0.99) -> dict[str, Any]:
    """Compute and verify the exact discounted Pareto leaf returns of fruit-tree-v0 by exact rollout enumeration."""
    env = make_mo_env("fruit-tree-v0")
    try:
        core: Any = env.unwrapped
        actual_depth = int(core.tree_depth)
        if actual_depth != tree_depth:
            raise ValueError(f"Expected fruit-tree-v0 depth {tree_depth}, got {actual_depth}")
        reward_dim = int(core.reward_dim)
        num_leaves = 1 << tree_depth
        leaf_returns = np.zeros((num_leaves, reward_dim), dtype=np.float64)
        raw_leaf_returns = np.zeros((num_leaves, reward_dim), dtype=np.float64)
        action_paths: list[list[int]] = []

        for leaf_idx in range(num_leaves):
            obs, _ = env.reset(seed=0)
            path = [(leaf_idx >> shift) & 1 for shift in reversed(range(tree_depth))]
            raw = np.zeros(reward_dim, dtype=np.float64)
            discounted = np.zeros(reward_dim, dtype=np.float64)
            discount = 1.0
            terminated = False
            for action in path:
                obs, reward, terminated, truncated, _ = env.step(action)
                r = np.asarray(reward, dtype=np.float64)
                raw += r
                discounted += discount * r
                discount *= gamma
                if truncated:
                    raise RuntimeError("Unexpected truncation in fruit-tree-v0")
            if not terminated or int(obs[0]) != tree_depth or int(obs[1]) != leaf_idx:
                raise RuntimeError(f"Path {path} did not terminate at expected leaf {leaf_idx}; obs={obs}")
            raw_leaf_returns[leaf_idx] = raw
            leaf_returns[leaf_idx] = discounted
            action_paths.append(path)

        env_pf = np.asarray(core.pareto_front(gamma), dtype=np.float64)
        max_abs_diff = float(np.max(np.abs(leaf_returns - env_pf)))
        if max_abs_diff > 1e-5:
            raise ValueError(f"Rollout oracle disagreed with env.pareto_front({gamma}): max_diff={max_abs_diff}")
    finally:
        env.close()

    return {
        "env_id": "fruit-tree-v0",
        "tree_depth": tree_depth,
        "reward_dim": reward_dim,
        "gamma": gamma,
        "num_leaves": num_leaves,
        "discounted_leaf_returns": leaf_returns,
        "raw_leaf_returns": raw_leaf_returns,
        "action_paths": action_paths,
        "verified_max_abs_diff_vs_env_pareto_front": max_abs_diff,
    }


def exact_oracle_utility(preferences: np.ndarray, discounted_leaf_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (optimal_utilities, optimal_leaf_indices) for each preference w in preferences."""
    weights = np.asarray(preferences, dtype=np.float64)
    leaves = np.asarray(discounted_leaf_returns, dtype=np.float64)
    scores = weights @ leaves.T
    best_leaves = np.argmax(scores, axis=1)
    best_utilities = scores[np.arange(len(weights)), best_leaves]
    return best_utilities, best_leaves

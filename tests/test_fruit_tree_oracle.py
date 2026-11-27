from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.envs.fruit_tree_oracle import encode_fruit_tree_node, exact_oracle_utility, extract_fruit_tree_oracle
from src.models.face_exposure_ppo import SimplexPPO
from src.models.onpolicy import ActorCritic


def test_fruit_tree_oracle_matches_env_pareto_front_exactly():
    oracle = extract_fruit_tree_oracle(tree_depth=6, gamma=0.99)
    assert oracle["num_leaves"] == 64
    assert oracle["reward_dim"] == 6
    assert oracle["verified_max_abs_diff_vs_env_pareto_front"] < 1e-6

    vertices = np.eye(6, dtype=np.float64)
    opt_u, opt_leaves = exact_oracle_utility(vertices, oracle["discounted_leaf_returns"])
    assert len(set(opt_leaves.tolist())) == 6
    assert np.all(opt_u > 7.5)


def test_fruit_tree_objective_subset_oracles_for_d_scaling_3_to_6():
    for dim in (3, 4, 5, 6):
        subset = extract_fruit_tree_oracle(tree_depth=6, gamma=0.99, objective_indices=range(dim))
        assert subset["reward_dim"] == dim
        assert subset["discounted_leaf_returns"].shape == (64, dim)
        assert subset["verified_max_abs_diff_vs_env_pareto_front"] < 1e-6
        _, opt_leaves = exact_oracle_utility(np.eye(dim, dtype=np.float64), subset["discounted_leaf_returns"])
        assert len(set(opt_leaves.tolist())) == dim


def test_encode_fruit_tree_node_is_bijective_one_hot():
    seen: set[int] = set()
    for row in range(7):
        for pos in range(1 << row):
            vec = encode_fruit_tree_node(np.array([row, pos], dtype=np.int32), tree_depth=6)
            assert vec.shape == (127,)
            assert float(vec.sum()) == pytest.approx(1.0)
            idx = int(np.argmax(vec))
            assert idx not in seen
            seen.add(idx)
    assert len(seen) == 127


def test_simplex_ppo_update_supports_binary_actions_and_six_objectives():
    torch.manual_seed(42)
    model = ActorCritic(state_dim=127, action_dim=2, reward_dim=6, width=32)
    plan = {
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "training_interactions": 1024,
        "epochs": 2,
        "minibatches": 2,
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
    }
    learner = SimplexPPO(model, plan)
    states = torch.zeros(6, 4, 127)
    states[:, :, 0] = 1.0
    prefs = torch.full((6, 4, 6), 1.0 / 6.0)
    with torch.no_grad():
        dist = model.distribution(states, prefs)
        actions = dist.sample()
        batch = {
            "states": states,
            "preferences": prefs,
            "actions": actions,
            "logp": dist.log_prob(actions),
            "values": model.value(states, prefs),
            "next_values": model.value(states, prefs),
            "rewards": torch.rand(6, 4, 6),
            "terminated": torch.zeros(6, 4, dtype=torch.bool),
            "truncated": torch.zeros(6, 4, dtype=torch.bool),
        }
    batch["terminated"][-1] = True
    loss = learner.update(batch, step=24)
    assert all(np.isfinite(list(loss.values())))
    assert loss["entropy_target"] <= math.log(2) + 1e-9
    assert learner.actor_updates == 4
    assert learner.critic_updates == 8

from __future__ import annotations

import numpy as np
import pytest

from src.envs.dst_oracle import encode_dst_state, extract_dst_oracle


def test_dst_oracle_matches_env_pareto_front_exactly():
    oracle = extract_dst_oracle(gamma=0.99)
    assert oracle["reward_dim"] == 2
    assert oracle["num_treasures"] == 10
    assert oracle["verified_max_abs_diff_vs_env_pareto_front"] < 1e-4


def test_encode_dst_state_is_bijective_one_hot():
    seen: set[int] = set()
    for r in range(11):
        for c in range(11):
            vec = encode_dst_state(np.array([r, c], dtype=np.int32))
            assert vec.shape == (121,)
            assert float(vec.sum()) == pytest.approx(1.0)
            idx = int(np.argmax(vec))
            assert idx not in seen
            seen.add(idx)
    assert len(seen) == 121

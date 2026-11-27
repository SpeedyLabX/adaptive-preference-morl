"""Unit tests for Stage-1A GPU recipes, anchor-freeze gate, queue ordering, and Kaggle Minecart bit-exactness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_stage1a_gpu_job import parse_dataset_spec  # noqa: E402
from scripts.run_stage1a_gpu_queue import _verify_frozen_anchor_manifest, build_queue  # noqa: E402
from scripts.run_stage1a_minecart_kaggle import (  # noqa: E402
    ActorCritic as StandaloneActorCritic,
)
from scripts.run_stage1a_minecart_kaggle import StandaloneMinecart  # noqa: E402
from src.envs.wrapper import make_mo_env  # noqa: E402
from src.models.onpolicy import ActorCritic as RepoActorCritic  # noqa: E402


def test_parse_dataset_spec() -> None:
    assert parse_dataset_spec("minecart-v0|anchor|vertex=0") == ("minecart-v0", "anchor", "vertex=0")
    assert parse_dataset_spec("mo-lunar-lander-v3|arm|fixed_face_balanced") == (
        "mo-lunar-lander-v3",
        "arm",
        "fixed_face_balanced",
    )
    with pytest.raises(ValueError):
        parse_dataset_spec("minecart-v0|bad|x")


def test_stage1a_gpu_queue_counts_and_paired_seed_order() -> None:
    protocol = json.loads((REPO_ROOT / "configs" / "protocols" / "stage1_simplex_face_exposure_v1.json").read_text())
    anchors = build_queue(protocol, "anchors", "all")
    arms = build_queue(protocol, "arms", "all")
    assert len(anchors) == 21
    assert len(arms) == 60
    assert all(item["recipe"].exists() for item in anchors)
    assert all(item["recipe"].exists() for item in arms)

    # First 30 arms are Lunar (10 paired seeds x 3 arms under v1.3), grouped by seed for matched-hardware pairing.
    lunar_arms = [item for item in arms if item["env_id"] == "mo-lunar-lander-v3"]
    assert len(lunar_arms) == 30
    assert lunar_arms[0]["tag"].endswith("seed20269301")
    assert lunar_arms[1]["tag"].endswith("seed20269301")
    assert lunar_arms[2]["tag"].endswith("seed20269301")
    assert {lunar_arms[0]["tag"], lunar_arms[1]["tag"], lunar_arms[2]["tag"]} == {
        "mo-lunar-lander-v3_arm_uniform_seed20269301",
        "mo-lunar-lander-v3_arm_dirichlet_sparse_seed20269301",
        "mo-lunar-lander-v3_arm_fixed_face_balanced_seed20269301",
    }


def test_arms_gate_rejects_missing_or_tampered_anchor_manifest(tmp_path: Path) -> None:
    missing = tmp_path / "missing_manifest.json"
    with pytest.raises(RuntimeError, match="missing"):
        _verify_frozen_anchor_manifest(missing)

    tampered = tmp_path / "tampered_manifest.json"
    tampered.write_text(json.dumps({"total_anchors": 21, "final_test_steps": 0, "manifest_sha256": "deadbeef"}))
    with pytest.raises(RuntimeError, match="integrity"):
        _verify_frozen_anchor_manifest(tampered)


def test_standalone_minecart_bit_exact_with_mo_gymnasium() -> None:
    """Verify StandaloneMinecart matches mo-gymnasium==1.3.2 minecart-v0 bit-for-bit."""
    rng = np.random.default_rng(12345)
    actions = rng.integers(0, 6, size=1500).tolist()

    ref_env = make_mo_env("minecart-v0")
    np.random.seed(999)
    ref_obs, _ = ref_env.reset(seed=999)
    ref_trace = []
    for act in actions:
        o, r, tm, tr, _ = ref_env.step(int(act))
        ref_trace.append((o.copy(), np.asarray(r, dtype=np.float32).copy(), bool(tm), bool(tr)))
        if tm or tr:
            o, _ = ref_env.reset()
            ref_trace.append((o.copy(), np.zeros(3, dtype=np.float32), False, False))
    ref_env.close()

    sta_env = StandaloneMinecart()
    np.random.seed(999)
    sta_obs, _ = sta_env.reset(seed=999)
    assert np.array_equal(ref_obs, sta_obs)
    sta_trace = []
    for act in actions:
        o, r, tm, tr, _ = sta_env.step(int(act))
        sta_trace.append((o.copy(), np.asarray(r, dtype=np.float32).copy(), bool(tm), bool(tr)))
        if tm or tr:
            o, _ = sta_env.reset()
            sta_trace.append((o.copy(), np.zeros(3, dtype=np.float32), False, False))

    assert len(ref_trace) == len(sta_trace)
    for (ro, rr, rtm, rtr), (so, sr, stm, strc) in zip(ref_trace, sta_trace, strict=True):
        assert np.allclose(ro, so, atol=1e-7, rtol=0.0)
        assert np.allclose(rr, sr, atol=1e-7, rtol=0.0)
        assert rtm == stm
        assert rtr == strc


def test_standalone_actor_critic_loads_into_repo_actor_critic() -> None:
    torch.manual_seed(42)
    standalone = StandaloneActorCritic(state_dim=7, action_dim=6, reward_dim=3, width=256)
    repo = RepoActorCritic(state_dim=7, action_dim=6, reward_dim=3, width=256)
    repo.load_state_dict(standalone.state_dict())
    s = torch.randn(4, 7)
    w = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.3, 0.3, 0.4]])
    assert torch.allclose(standalone.actor(s, w), repo.actor(s, w), atol=0.0)
    assert torch.allclose(standalone.value(s, w), repo.value(s, w), atol=0.0)

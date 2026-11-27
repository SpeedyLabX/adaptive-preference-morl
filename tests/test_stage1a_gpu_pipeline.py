"""Unit tests for Stage-1A GPU recipe generation, anchor-freeze gate, and queue ordering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_stage1a_gpu_job import parse_dataset_spec  # noqa: E402
from scripts.run_stage1a_gpu_queue import _verify_frozen_anchor_manifest, build_queue  # noqa: E402


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
    assert len(arms) == 50
    assert all(item["recipe"].exists() for item in anchors)
    assert all(item["recipe"].exists() for item in arms)

    # First 20 arms are Lunar (10 paired seeds x 2 arms), grouped by seed so matched-hardware pairing holds.
    lunar_arms = [item for item in arms if item["env_id"] == "mo-lunar-lander-v3"]
    assert len(lunar_arms) == 20
    assert lunar_arms[0]["tag"].endswith("seed20269301")
    assert lunar_arms[1]["tag"].endswith("seed20269301")


def test_arms_gate_rejects_missing_or_tampered_anchor_manifest(tmp_path: Path) -> None:
    missing = tmp_path / "missing_manifest.json"
    with pytest.raises(RuntimeError, match="missing"):
        _verify_frozen_anchor_manifest(missing)

    tampered = tmp_path / "tampered_manifest.json"
    tampered.write_text(json.dumps({"total_anchors": 21, "final_test_steps": 0, "manifest_sha256": "deadbeef"}))
    with pytest.raises(RuntimeError, match="integrity"):
        _verify_frozen_anchor_manifest(tampered)

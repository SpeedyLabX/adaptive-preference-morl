from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROTOCOL_PATH = Path("configs/protocols/stage1_simplex_face_exposure_v1.json")


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_stage1_protocol_schema_locks_and_hashes():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "stage1-simplex-face-exposure-v1"
    assert protocol["status"] == "frozen_pending_gate1_audit"
    assert protocol["execution_authorized"] is False
    assert protocol["gate1_signoff_required"] is True
    assert protocol["learner_contract"]["hindsight_relabeling"] is False

    prereq = protocol["week1_gate_prerequisites"]
    assert (
        _sha256(prereq["step2_minecart_metric_differentiation"]["evidence"])
        == (prereq["step2_minecart_metric_differentiation"]["evidence_sha256"])
    )
    assert (
        _sha256(prereq["step2_fruit_tree_d6_pilot"]["evidence"])
        == (prereq["step2_fruit_tree_d6_pilot"]["evidence_sha256"])
    )
    assert Path(prereq["throughput_qualification"]["receipt"]).is_file()

    # Retention margin and adaptive rule invariants.
    margin = protocol["metrics_and_analysis"]["interior_utility_retention_margin"]
    assert margin["minimum_retention_ratio"] == 0.985
    assert margin["maximum_relative_degradation"] == 0.015
    assert protocol["arms"]["stage1b_conditional_adaptive_expansion"]["enabled_in_initial_sweep"] is False


def test_stage1_protocol_benchmarks_seeds_disjoint_and_budgets_verified():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    pilot_seeds = set(protocol["week1_gate_prerequisites"]["step2_fruit_tree_d6_pilot"]["pilot_seeds"])

    benchmarks = {b["env_id"]: b for b in protocol["benchmarks"]}
    assert set(benchmarks) == {"fruit-tree-v0", "minecart-v0", "mo-lunar-lander-v3", "deep-sea-treasure-v0"}

    core_dims = [b["reward_dim"] for b in benchmarks.values() if b["role"] == "core_d_scaling"]
    assert sorted(core_dims) == [3, 4, 6]
    assert benchmarks["deep-sea-treasure-v0"]["role"] == "sanity_check_d2_only"
    assert benchmarks["deep-sea-treasure-v0"]["excluded_from_d_ge_3_starvation_claim"] is True

    # Verify exact 21 budgeted vertex-specialist anchors (9 Minecart + 12 Lunar).
    total_anchors = sum(int(b["anchor_runs_budgeted"]) for b in benchmarks.values())
    assert total_anchors == 21
    assert protocol["compute_budget_and_routing"]["stage1a_run_counts"]["vertex_specialist_anchors_gpu"] == 21

    # Verify all training seeds are >= 10, unique, disjoint from pilot seeds, and dev/test split seeds are disjoint.
    all_train_seeds: set[int] = set()
    all_eval_seeds: set[int] = set()
    for b in benchmarks.values():
        seeds = b["paired_seeds"]
        assert 10 <= len(seeds) <= 12
        assert len(set(seeds)) == len(seeds)
        assert not (set(seeds) & pilot_seeds)
        assert not (set(seeds) & all_train_seeds)
        all_train_seeds.update(seeds)

        assert b["verified_max_gpu_hours_per_run"] <= 2.0
        dev_seed = b["evaluation_splits"]["development_seed"]
        test_seed = b["evaluation_splits"]["final_test_seed"]
        assert dev_seed != test_seed
        assert dev_seed not in all_eval_seeds and test_seed not in all_eval_seeds
        all_eval_seeds.update({dev_seed, test_seed})

        if "base_protocol" in b:
            assert _sha256(b["base_protocol"]) == b["base_protocol_sha256"]
        if "reference_front_artifact" in b:
            assert _sha256(b["reference_front_artifact"]) == b["reference_front_sha256"]

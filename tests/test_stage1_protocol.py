from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROTOCOL_PATH = Path("configs/protocols/stage1_simplex_face_exposure_v1.json")


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_stage1_protocol_schema_locks_and_gate1_fixes():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "stage1-simplex-face-exposure-v1"
    assert protocol["status"] == "frozen_gate1b_passed_authorized"
    assert protocol["execution_authorized"] is True
    assert protocol["gate1_signoff_required"] is True
    assert protocol["learner_contract"]["hindsight_relabeling"] is False

    # Fix 6: Intervention-only claim adjudication.
    adj = protocol["claim_scope_adjudication"]
    assert adj["intervention_only_novelty"] is True
    assert adj["metric_novelty_dropped"] is True

    # Prerequisite artifact hashes and PD-MORL full-text check.
    prereq = protocol["week1_gate_prerequisites"]
    step1_ids = {p["paper_id"] for p in prereq["step1_full_text_literature_gate"]["papers"]}
    assert step1_ids == {"delasheras2026controllability", "bonin2026coverage-collapse", "basaklar2023pdmorl"}
    assert (
        _sha256(prereq["step2_minecart_metric_differentiation"]["evidence"])
        == (prereq["step2_minecart_metric_differentiation"]["evidence_sha256"])
    )
    assert (
        _sha256(prereq["step2_fruit_tree_d6_pilot"]["evidence"])
        == (prereq["step2_fruit_tree_d6_pilot"]["evidence_sha256"])
    )
    assert Path(prereq["throughput_qualification"]["receipt"]).is_file()
    assert Path(prereq["gate1_audit_receipt"]).is_file()
    assert Path(prereq["gate1b_reverification_receipt"]).is_file()
    assert prereq["gate1b_verdict"] == "PASS"

    # Fix 1: Strict retention criterion (no OR clause) + reference-shifted positive denominator.
    margin = protocol["metrics_and_analysis"]["interior_utility_retention_margin"]
    assert margin["informational_equivalent_ratio"] == 0.985
    assert margin["maximum_relative_degradation"] == 0.015
    assert " OR " not in margin["sole_criterion"]
    assert margin["denominator_normalization"] == "reference_shifted_positive_utility_U_minus_wTr_ref"

    # Gate-1B (a): Frozen d-scaling OLS slope test statistic.
    d_stat = protocol["fruit_tree_d_subset_scaling_sweep"]["d_scaling_test_statistic"]
    assert "OLS slope beta_d" in d_stat["test_statistic"]
    assert "beta_d > 0" in d_stat["hypothesis_test"]

    # Fix 2: Single primary endpoint per benchmark (exact vertices excluded) and Holm k=3.
    endpoint = protocol["metrics_and_analysis"]["single_primary_endpoint_per_benchmark"]
    assert endpoint["name"] == "argmax_deployed_non_vertex_face_cap_mean_regret"
    assert endpoint["exact_vertices_excluded"] is True
    assert endpoint["holm_family_size"] == 3
    assert endpoint["holm_family_benchmarks"] == ["fruit-tree-v0", "minecart-v0", "mo-lunar-lander-v3"]

    # Fix 4 & 5: Dirichlet sparse control present, adaptive locked, Minecart 2x2 explicitly deferred.
    assert protocol["arms"]["stage1a_sparse_prior_control"]["id"] == "dirichlet_sparse"
    assert protocol["arms"]["stage1b_conditional_adaptive_expansion"]["enabled_in_initial_sweep"] is False
    ctrl_2x2 = protocol["arms"]["secondary_minecart_2x2_normalization_control"]
    assert ctrl_2x2["status"] == "deferred_to_stage1b_or_appendix"
    assert ctrl_2x2["included_in_stage1a_budget"] is False


def test_stage1_protocol_benchmarks_d_subsets_seeds_disjoint_and_budgets_verified():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    pilot_seeds = set(protocol["week1_gate_prerequisites"]["step2_fruit_tree_d6_pilot"]["pilot_seeds"])

    benchmarks = {b["env_id"]: b for b in protocol["benchmarks"]}
    assert set(benchmarks) == {"fruit-tree-v0", "minecart-v0", "mo-lunar-lander-v3", "deep-sea-treasure-v0"}

    core_dims = [b["reward_dim"] for b in benchmarks.values() if b["role"] == "core_d_scaling"]
    assert sorted(core_dims) == [3, 4, 6]
    assert benchmarks["deep-sea-treasure-v0"]["role"] == "sanity_check_d2_only"
    assert benchmarks["deep-sea-treasure-v0"]["excluded_from_d_ge_3_starvation_claim"] is True

    # Fix 3: Approximate-reference benchmarks use unclamped signed gap + frozen anchors before eval.
    for approx_env in ("minecart-v0", "mo-lunar-lander-v3"):
        assert benchmarks[approx_env]["regret_estimator"] == "unclamped_signed_gap_primary_plus_fraction_clipped"
        assert (
            benchmarks[approx_env]["anchor_spec"]["selection_rule"]
            == "final_fixed_budget_checkpoint_only_frozen_and_hashed_before_arm_eval"
        )

    # Verify exact 21 budgeted vertex-specialist anchors (9 Minecart + 12 Lunar) and GPU-hour budget arithmetic.
    total_anchors = sum(int(b["anchor_runs_budgeted"]) for b in benchmarks.values())
    assert total_anchors == 21
    counts = protocol["compute_budget_and_routing"]["stage1a_run_counts"]
    assert counts["vertex_specialist_anchors_gpu"] == 21
    assert counts["total_stage1a_gpu_runs"] == (
        counts["minecart_d3_gpu_3arms"] + counts["lunar_d4_gpu_2arms"] + counts["vertex_specialist_anchors_gpu"]
    )
    assert counts["total_stage1a_cpu_runs"] == (
        counts["fruit_tree_d6_core_cpu"] + counts["fruit_tree_d3_to_d5_subset_cpu"] + counts["dst_d2_sanity_cpu"]
    )
    assert counts["total_stage1a_runs"] == counts["total_stage1a_cpu_runs"] + counts["total_stage1a_gpu_runs"]
    gpu_h = protocol["compute_budget_and_routing"]["estimated_stage1a_gpu_hours"]
    assert (
        round(gpu_h["minecart_30_runs"] + gpu_h["lunar_20_runs"] + gpu_h["vertex_anchors_21_runs"], 1)
        == (gpu_h["total_gpu_hours"])
    )

    # Verify all training seeds (core + d-subsets + anchors) are disjoint from pilot seeds and each other.
    all_train_seeds: set[int] = set()
    all_eval_seeds: set[int] = set()
    for b in benchmarks.values():
        seeds = b["paired_seeds"]
        assert 10 <= len(seeds) <= 12
        assert len(set(seeds)) == len(seeds)
        assert not (set(seeds) & pilot_seeds)
        assert not (set(seeds) & all_train_seeds)
        all_train_seeds.update(seeds)
        if "anchor_spec" in b:
            anchor_seeds = set(b["anchor_spec"]["anchor_seeds"])
            assert not (anchor_seeds & all_train_seeds)
            all_train_seeds.update(anchor_seeds)

        max_hours = b.get("verified_max_gpu_hours_per_run", b.get("verified_max_hours_per_run"))
        assert max_hours is not None and max_hours <= 2.0
        comp = b["evaluation_splits"]["primary_non_vertex_face_cap_composition"]
        assert comp["exact_vertices_excluded"] is True
        dev_seed = b["evaluation_splits"]["development_seed"]
        test_seed = b["evaluation_splits"]["final_test_seed"]
        assert dev_seed != test_seed
        assert dev_seed not in all_eval_seeds and test_seed not in all_eval_seeds
        all_eval_seeds.update({dev_seed, test_seed})

        if "base_protocol" in b:
            assert _sha256(b["base_protocol"]) == b["base_protocol_sha256"]
        if "reference_front_artifact" in b:
            assert _sha256(b["reference_front_artifact"]) == b["reference_front_sha256"]

    # Fix 4: Verify fruit-tree d in {3, 4, 5, 6} subset sweep seeds and dimensions.
    sweep = protocol["fruit_tree_d_subset_scaling_sweep"]
    assert [s["reward_dim"] for s in sweep["subsets"]] == [3, 4, 5, 6]
    for sub in sweep["subsets"]:
        if sub.get("reuses_core_benchmark_runs"):
            continue
        sub_seeds = set(sub["paired_seeds"])
        assert len(sub_seeds) == 10
        assert not (sub_seeds & pilot_seeds)
        assert not (sub_seeds & all_train_seeds)
        all_train_seeds.update(sub_seeds)
        assert sub["development_seed"] not in all_eval_seeds
        assert sub["final_test_seed"] not in all_eval_seeds
        all_eval_seeds.update({sub["development_seed"], sub["final_test_seed"]})

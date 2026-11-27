from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.simplex_metrics import (
    FaceBalancedPreferenceSampler,
    compute_abandoned_preference_mass,
    compute_checkpoint_metrics_bundle,
    compute_effective_coverage,
    compute_rank_correlations,
    sample_edge_cap_preferences,
    sample_uniform_simplex,
    sample_vertex_cap_preferences,
)


def test_apm_negative_utility_sign_guard_prevents_baseline_self_abandonment():
    # Fuel-only negative returns mimicking Minecart cost-heavy preferences.
    baseline_front = np.array(
        [
            [0.0, 0.0, -0.25],
            [1.0, 0.0, -0.90],
            [0.0, 1.0, -0.90],
        ],
        dtype=np.float64,
    )
    preferences = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float64,
    )
    reference_point = np.array([-0.1, -0.1, -1.2], dtype=np.float64)

    raw = compute_abandoned_preference_mass(
        baseline_front, baseline_front, preferences, tau=0.1, sign_guard="raw_unguarded"
    )
    assert raw["negative_baseline_count"] == 2
    assert raw["apm"] == pytest.approx(0.5)  # Raw formula falsely flags negative-utility baseline points.

    mag = compute_abandoned_preference_mass(
        baseline_front, baseline_front, preferences, tau=0.1, sign_guard="magnitude"
    )
    assert mag["apm"] == pytest.approx(0.0)
    assert mag["abandoned_count"] == 0

    shifted = compute_abandoned_preference_mass(
        baseline_front,
        baseline_front,
        preferences,
        tau=0.1,
        sign_guard="reference_shifted",
        reference_point=reference_point,
    )
    assert shifted["apm"] == pytest.approx(0.0)
    assert shifted["abandoned_count"] == 0


def test_apm_detects_worse_negative_utility_under_magnitude_guard():
    baseline_front = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
    worse_front = np.array([[0.0, 0.0, -1.25]], dtype=np.float64)
    better_front = np.array([[0.0, 0.0, -0.80]], dtype=np.float64)
    preferences = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

    worse_res = compute_abandoned_preference_mass(
        worse_front, baseline_front, preferences, tau=0.1, sign_guard="magnitude"
    )
    better_res = compute_abandoned_preference_mass(
        better_front, baseline_front, preferences, tau=0.1, sign_guard="magnitude"
    )
    assert worse_res["apm"] == pytest.approx(1.0)
    assert better_res["apm"] == pytest.approx(0.0)


def test_effective_coverage_and_archive_blindness_gap():
    preferences = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    reference = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    # Inverted conditioning: policy produces the exact Pareto front, but swapped for the requested weights!
    deployed_returns = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64)

    bundle = compute_checkpoint_metrics_bundle(deployed_returns, preferences, reference)
    assert bundle["effective_coverage"] == pytest.approx(2.0)
    assert bundle["apm_magnitude"]["apm"] == pytest.approx(0.0)
    assert bundle["deployed_regret"]["mean_regret"] == pytest.approx(2.0)
    assert bundle["deployed_regret"]["archive_blindness_gap"] == pytest.approx(2.0)
    assert bundle["controllability"]["mean_spearman"] == pytest.approx(-1.0)
    assert compute_effective_coverage(deployed_returns, preferences) == pytest.approx(2.0)


def test_simplex_cap_generators_and_face_balanced_sampler_bounds():
    rng = np.random.default_rng(123)
    unif = sample_uniform_simplex(6, 50, rng)
    assert unif.shape == (50, 6)
    assert np.allclose(unif.sum(axis=1), 1.0, atol=1e-5)

    vcaps = sample_vertex_cap_preferences(6, per_vertex=5, epsilon=0.10, rng=rng)
    assert vcaps.shape == (30, 6)
    assert np.allclose(vcaps.sum(axis=1), 1.0, atol=1e-5)
    for vertex in range(6):
        block = vcaps[vertex * 5 : (vertex + 1) * 5]
        assert np.all(block[:, vertex] >= 0.9 - 1e-5)

    ecaps = sample_edge_cap_preferences(6, per_edge=2, epsilon=0.10, rng=rng)
    assert ecaps.shape == (30, 6)  # C(6, 2) = 15 edges * 2
    assert np.allclose(ecaps.sum(axis=1), 1.0, atol=1e-5)

    sampler = FaceBalancedPreferenceSampler(6, "fixed_face_balanced", seed=7, interior_prob=0.5, vertex_prob=0.25)
    draws = np.stack([sampler.sample() for _ in range(2000)])
    assert draws.shape == (2000, 6)
    assert np.allclose(draws.sum(axis=1), 1.0, atol=1e-5)
    summary = sampler.exposure_summary()
    assert summary["total_draws"] == 2000
    assert 0.20 <= summary["stratum_fractions"]["vertex"] <= 0.30
    assert 0.45 <= summary["stratum_fractions"]["interior"] <= 0.55
    assert all(count > 40 for count in summary["vertex_counts"])


def test_rank_correlations_detect_non_identical_checkpoint_rankings():
    correlations = compute_rank_correlations(
        {
            "neg_face_cap_regret": [1.0, 2.0, 3.0, 4.0, 5.0],
            "mean_spearman": [5.0, 1.0, 4.0, 2.0, 3.0],
            "neg_apm": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    assert correlations["spearman"]["neg_face_cap_regret"]["neg_apm"] == pytest.approx(1.0)
    assert correlations["spearman"]["neg_face_cap_regret"]["mean_spearman"] < 0.0

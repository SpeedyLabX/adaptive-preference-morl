"""Simplex face-cap evaluation metrics, APM (sign-guarded), and static face-balanced sampling.

Implements:
- Effective Coverage C and Abandoned Preference Mass (APM) from Bonin et al. (2026, arXiv:2609.26918),
  with explicit sign guards for benchmarks whose scalarized utility can be negative.
- Deployed scalarized regret vs. archive/front utility (exposing archive blindness).
- Exact simplex vertex/edge epsilon-cap split generators and static Fixed Face-Balanced sampler.
- Pairwise Spearman/Kendall checkpoint-ranking correlation checks for Week-1 metric differentiation.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import kendalltau

from src.evaluation.metrics import (
    _safe_rank_correlation,
    compute_per_objective_controllability,
    pareto_prune,
)
from src.evaluation.protocol import validate_preferences


def compute_front_utilities(front_points: np.ndarray, preferences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (max_utilities, argmax_indices) over a Pareto-pruned front for each preference."""
    frontier = pareto_prune(front_points).astype(np.float64)
    if frontier.size == 0:
        raise ValueError("front_points must contain at least one finite point")
    weights = validate_preferences(preferences, frontier.shape[1]).astype(np.float64)
    scores = weights @ frontier.T  # [M, |F|]
    best_indices = np.argmax(scores, axis=1)
    best_utilities = scores[np.arange(len(weights)), best_indices]
    return best_utilities, best_indices


def compute_effective_coverage(front_points: np.ndarray, preferences: np.ndarray) -> float:
    """Effective Coverage C(F) = exp(-sum_k rho_k ln rho_k) (Bonin et al., 2026, Eq. 1)."""
    frontier = pareto_prune(front_points).astype(np.float64)
    if frontier.size == 0:
        return 0.0
    _, best_indices = compute_front_utilities(frontier, preferences)
    counts = np.bincount(best_indices, minlength=len(frontier)).astype(np.float64)
    probs = counts[counts > 0.0] / float(len(best_indices))
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(np.exp(entropy))


def compute_abandoned_preference_mass(
    front_points: np.ndarray,
    baseline_front: np.ndarray,
    preferences: np.ndarray,
    tau: float = 0.1,
    sign_guard: str = "magnitude",
    reference_point: np.ndarray | None = None,
    zero_tol: float = 1e-12,
) -> dict[str, Any]:
    """Compute Abandoned Preference Mass (Bonin et al., 2026, Eq. 2) with negative-utility sign guards.

    Raw Bonin Eq. 2 checks U_F(w) < (1 - tau) * U_{F0}(w). When U_{F0}(w) < 0 (e.g., negative cost/fuel
    objectives in minecart-v0), (1 - tau) * U_{F0}(w) > U_{F0}(w), which reverses the inequality and
    falsely flags even the baseline front F0 itself as abandoned.

    Supported sign_guard modes:
    - "magnitude" (default): U_F(w) < U_{F0}(w) - tau * |U_{F0}(w)| (with -zero_tol when |U_{F0}(w)| <= zero_tol).
      Identical to Bonin Eq. 2 whenever U_{F0}(w) > 0, and guarantees APM(F0; F0) == 0 for all signs.
    - "reference_shifted": shifts both utilities by -w^T r_ref > 0 using a strictly dominated reference_point,
      then applies U_tilde_F(w) < (1 - tau) * U_tilde_{F0}(w).
    - "raw_unguarded": literal U_F(w) < (1 - tau) * U_{F0}(w), retained only to audit the sign-flip edge case.
    """
    if not 0.0 <= tau < 1.0:
        raise ValueError("tau must lie in [0, 1)")
    if sign_guard not in {"magnitude", "reference_shifted", "raw_unguarded"}:
        raise ValueError(f"Unsupported sign_guard mode: {sign_guard}")

    front_u, _ = compute_front_utilities(front_points, preferences)
    base_u, _ = compute_front_utilities(baseline_front, preferences)
    weights = np.asarray(preferences, dtype=np.float64)
    negative_baseline_count = int(np.sum(base_u < -zero_tol))

    if sign_guard == "magnitude":
        near_zero = np.abs(base_u) <= zero_tol
        threshold = np.where(near_zero, -zero_tol, base_u - tau * np.abs(base_u))
        abandoned = front_u < threshold
    elif sign_guard == "reference_shifted":
        if reference_point is None:
            raise ValueError("reference_point is required when sign_guard='reference_shifted'")
        ref = np.asarray(reference_point, dtype=np.float64)
        if ref.shape != (weights.shape[1],):
            raise ValueError(f"Expected reference_point of shape ({weights.shape[1]},), got {ref.shape}")
        ref_u = weights @ ref
        shifted_base = base_u - ref_u
        if np.any(shifted_base <= zero_tol):
            raise ValueError("baseline_front must strictly exceed reference_point scalarized utility for all weights")
        shifted_front = front_u - ref_u
        abandoned = shifted_front < (1.0 - tau) * shifted_base
    else:
        abandoned = front_u < (1.0 - tau) * base_u

    abandoned_count = int(np.sum(abandoned))
    total = len(weights)
    return {
        "apm": float(abandoned_count / total),
        "abandoned_count": abandoned_count,
        "total_preferences": total,
        "negative_baseline_count": negative_baseline_count,
        "tau": float(tau),
        "sign_guard": sign_guard,
        "mean_front_utility": float(np.mean(front_u)),
        "baseline_mean_front_utility": float(np.mean(base_u)),
    }


def compute_deployed_regret(
    deployed_returns: np.ndarray,
    reference_front: np.ndarray,
    preferences: np.ndarray,
) -> dict[str, float]:
    """Compute deployed single-query scalarized regret against a reference/oracle front.

    Also reports archive/front utility computed from the policy's own achieved return cloud so
    Archive Blindness (archive utility - deployed utility) is directly visible.
    """
    returns = np.asarray(deployed_returns, dtype=np.float64)
    reference = np.asarray(reference_front, dtype=np.float64)
    if returns.ndim != 2 or reference.ndim != 2 or returns.shape[1] != reference.shape[1]:
        raise ValueError("deployed_returns and reference_front must be 2D with matching reward_dim")
    weights = validate_preferences(preferences, returns.shape[1]).astype(np.float64)
    if len(returns) != len(weights):
        raise ValueError("deployed_returns and preferences must have the same length")

    oracle_u = np.max(weights @ reference.T, axis=1)
    deployed_u = np.sum(weights * returns, axis=1)
    archive_u, _ = compute_front_utilities(returns, weights)

    signed_gap = oracle_u - deployed_u
    regrets = np.maximum(0.0, signed_gap)
    tail_size = max(1, int(np.ceil(0.1 * len(regrets))))
    cvar_10 = float(np.sort(regrets)[-tail_size:].mean())
    return {
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "worst_decile_regret": float(np.quantile(regrets, 0.9)),
        "cvar_10_regret": cvar_10,
        "max_regret": float(np.max(regrets)),
        "mean_signed_gap": float(np.mean(signed_gap)),
        "mean_deployed_utility": float(np.mean(deployed_u)),
        "mean_oracle_utility": float(np.mean(oracle_u)),
        "mean_archive_utility": float(np.mean(archive_u)),
        "archive_blindness_gap": float(np.mean(archive_u - deployed_u)),
    }


def compute_rank_correlations(
    higher_is_better_scores: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute pairwise tie-aware Spearman rho and Kendall tau-b across checkpoint metrics."""
    names = list(higher_is_better_scores)
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in higher_is_better_scores.items()}
    lengths = {len(arr) for arr in arrays.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("All metric series must have identical length >= 2")

    spearman: dict[str, dict[str, float]] = {name: {} for name in names}
    kendall: dict[str, dict[str, float]] = {name: {} for name in names}
    for first in names:
        for second in names:
            spearman[first][second] = float(_safe_rank_correlation(arrays[first], arrays[second]))
            tau_val = kendalltau(arrays[first], arrays[second]).statistic
            kendall[first][second] = 0.0 if np.isnan(tau_val) else float(tau_val)
    return {"spearman": spearman, "kendall_tau_b": kendall}


def compute_checkpoint_metrics_bundle(
    deployed_returns: np.ndarray,
    preferences: np.ndarray,
    reference_front: np.ndarray,
    reference_point: np.ndarray | None = None,
    tau: float = 0.1,
) -> dict[str, Any]:
    """Compute Spearman (2605.10585), sign-guarded APM & C (2609.26918), and deployed regret together."""
    controllability = compute_per_objective_controllability(deployed_returns, preferences)
    regret = compute_deployed_regret(deployed_returns, reference_front, preferences)
    apm_magnitude = compute_abandoned_preference_mass(
        deployed_returns,
        reference_front,
        preferences,
        tau=tau,
        sign_guard="magnitude",
    )
    apm_raw = compute_abandoned_preference_mass(
        deployed_returns,
        reference_front,
        preferences,
        tau=tau,
        sign_guard="raw_unguarded",
    )
    bundle: dict[str, Any] = {
        "controllability": controllability,
        "deployed_regret": regret,
        "effective_coverage": compute_effective_coverage(deployed_returns, preferences),
        "apm_magnitude": apm_magnitude,
        "apm_raw_unguarded": apm_raw,
    }
    if reference_point is not None:
        bundle["apm_reference_shifted"] = compute_abandoned_preference_mass(
            deployed_returns,
            reference_front,
            preferences,
            tau=tau,
            sign_guard="reference_shifted",
            reference_point=reference_point,
        )
    return bundle


def sample_uniform_simplex(reward_dim: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample count preferences uniformly from Delta^{d-1} via Dirichlet(1_d)."""
    if reward_dim < 2 or count <= 0:
        raise ValueError("reward_dim must be >= 2 and count must be positive")
    return rng.dirichlet(np.ones(reward_dim, dtype=np.float64), size=count).astype(np.float32)


def sample_vertex_cap_preferences(
    reward_dim: int,
    per_vertex: int,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample exactly uniformly on each vertex epsilon-cap {w in Delta^{d-1} : w_i >= 1 - epsilon}."""
    if reward_dim < 2 or per_vertex <= 0:
        raise ValueError("reward_dim must be >= 2 and per_vertex must be positive")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly in (0, 1)")

    rows: list[np.ndarray] = []
    other_alpha = np.ones(reward_dim - 1, dtype=np.float64)
    for vertex in range(reward_dim):
        # Exact uniform volume density on the (d-1)-cap: remaining mass r = 1 - w_i ~ epsilon * U^{1/(d-1)}.
        u = rng.uniform(0.0, 1.0, size=per_vertex)
        remaining = epsilon * np.power(u, 1.0 / float(reward_dim - 1))
        other_shares = rng.dirichlet(other_alpha, size=per_vertex)
        for idx in range(per_vertex):
            w = np.zeros(reward_dim, dtype=np.float64)
            w[vertex] = 1.0 - remaining[idx]
            other_indices = [j for j in range(reward_dim) if j != vertex]
            w[other_indices] = remaining[idx] * other_shares[idx]
            w /= w.sum()
            rows.append(w)
    return np.asarray(rows, dtype=np.float32)


def sample_edge_cap_preferences(
    reward_dim: int,
    per_edge: int,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample near-edge (2-face) preferences {w in Delta^{d-1} : w_i + w_j >= 1 - epsilon} for d >= 3."""
    if reward_dim < 3 or per_edge <= 0:
        raise ValueError("edge-cap sampling requires reward_dim >= 3 and per_edge > 0")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly in (0, 1)")

    rows: list[np.ndarray] = []
    off_alpha = np.ones(reward_dim - 2, dtype=np.float64)
    for first, second in itertools.combinations(range(reward_dim), 2):
        off_indices = [j for j in range(reward_dim) if j not in (first, second)]
        # Keep both edge coordinates active (away from exact vertices) while staying within epsilon of the 1-face.
        edge_split = rng.uniform(0.15, 0.85, size=per_edge)
        off_mass = epsilon * np.power(rng.uniform(0.0, 1.0, size=per_edge), 1.0 / float(reward_dim - 2))
        off_shares = rng.dirichlet(off_alpha, size=per_edge)
        for idx in range(per_edge):
            w = np.zeros(reward_dim, dtype=np.float64)
            on_mass = 1.0 - off_mass[idx]
            w[first] = on_mass * edge_split[idx]
            w[second] = on_mass * (1.0 - edge_split[idx])
            w[off_indices] = off_mass[idx] * off_shares[idx]
            w /= w.sum()
            rows.append(w)
    return np.asarray(rows, dtype=np.float32)


class FaceBalancedPreferenceSampler:
    """Static (outcome-independent) simplex-face-balanced vs. uniform preference sampler.

    Modes:
    - "uniform": i.i.d. Dirichlet(1_d) over Delta^{d-1}.
    - "fixed_face_balanced": static mixture of
        * interior_prob on full simplex Dirichlet(1_d),
        * vertex_prob uniformly across the d simplex vertices (0-faces),
        * (1 - interior_prob - vertex_prob) stratified uniformly across proper sparse faces of
          support size s in {2, ..., d - 1} (for d >= 3).
      Zero outcome feedback, zero curriculum evaluation overhead.
    """

    def __init__(
        self,
        reward_dim: int,
        mode: str,
        seed: int,
        interior_prob: float = 0.50,
        vertex_prob: float = 0.25,
    ) -> None:
        if reward_dim < 2:
            raise ValueError("reward_dim must be >= 2")
        if mode not in {"uniform", "fixed_face_balanced"}:
            raise ValueError(f"Unsupported sampler mode: {mode}")
        if not 0.0 <= interior_prob <= 1.0 or not 0.0 <= vertex_prob <= 1.0:
            raise ValueError("Mixture probabilities must lie in [0, 1]")
        if interior_prob + vertex_prob > 1.0 + 1e-12:
            raise ValueError("interior_prob + vertex_prob cannot exceed 1.0")
        if reward_dim == 2 and mode == "fixed_face_balanced" and interior_prob + vertex_prob < 1.0 - 1e-12:
            raise ValueError("For d=2 there are no proper sparse faces of size 2..d-1; interior+vertex must sum to 1")

        self.reward_dim = reward_dim
        self.mode = mode
        self.interior_prob = float(interior_prob)
        self.vertex_prob = float(vertex_prob)
        self.sparse_face_prob = max(0.0, 1.0 - self.interior_prob - self.vertex_prob)
        self.rng = np.random.default_rng(seed)
        self.stratum_counts: dict[str, int] = {"interior": 0, "vertex": 0, "sparse_face": 0}
        self.vertex_counts = np.zeros(reward_dim, dtype=np.int64)
        self.support_size_counts = np.zeros(reward_dim + 1, dtype=np.int64)

    def sample(self) -> np.ndarray:
        if self.mode == "uniform":
            w = self.rng.dirichlet(np.ones(self.reward_dim, dtype=np.float64))
            self.stratum_counts["interior"] += 1
            self.support_size_counts[self.reward_dim] += 1
            return w.astype(np.float32)

        draw = float(self.rng.random())
        if draw < self.interior_prob:
            w = self.rng.dirichlet(np.ones(self.reward_dim, dtype=np.float64))
            self.stratum_counts["interior"] += 1
            self.support_size_counts[self.reward_dim] += 1
            return w.astype(np.float32)

        if draw < self.interior_prob + self.vertex_prob:
            vertex = int(self.rng.integers(self.reward_dim))
            w = np.zeros(self.reward_dim, dtype=np.float32)
            w[vertex] = 1.0
            self.stratum_counts["vertex"] += 1
            self.vertex_counts[vertex] += 1
            self.support_size_counts[1] += 1
            return w

        support_size = int(self.rng.integers(2, self.reward_dim))
        subset = self.rng.choice(self.reward_dim, size=support_size, replace=False)
        w = np.zeros(self.reward_dim, dtype=np.float64)
        w[subset] = self.rng.dirichlet(np.ones(support_size, dtype=np.float64))
        w /= w.sum()
        self.stratum_counts["sparse_face"] += 1
        self.support_size_counts[support_size] += 1
        return w.astype(np.float32)

    def exposure_summary(self) -> dict[str, Any]:
        total = sum(self.stratum_counts.values())
        return {
            "mode": self.mode,
            "total_draws": total,
            "stratum_counts": dict(self.stratum_counts),
            "stratum_fractions": {k: (v / total if total else 0.0) for k, v in self.stratum_counts.items()},
            "vertex_counts": self.vertex_counts.tolist(),
            "support_size_counts": {str(s): int(c) for s, c in enumerate(self.support_size_counts) if s >= 1},
        }

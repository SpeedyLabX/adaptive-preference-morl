# Gate-1 Pre-Sweep Protocol, Novelty & Scientific Code Audit (v1)

- **Reviewer:** Claude Code (**Claude Opus 5.5**, read-only via Windows `claude.exe -p --allowedTools "Read"`)
- **Date:** 2026-11-27
- **Audited Branch / Commit:** `research/week1-gate-step2` @ `010e6fa`
- **Audited Scope (Read-Only):**
  1. `docs/research/GATE1_AUDIT_PACKET_V1.md`
  2. `configs/protocols/stage1_simplex_face_exposure_v1.json`
  3. `src/evaluation/simplex_metrics.py`
  4. `research/evidence/week1_fruit_tree_pilot_v1.json` (summary / `paired_summary`)
  5. `research/claims.json`
- **Verdict:** **CONDITIONAL PASS** (6 blocking fixes required before setting `execution_authorized: true`)

---

## Verbatim Claude Opus 5.5 Audit Response

## Gate-1 audit: `stage1-simplex-face-exposure-v1` @ 010e6fa

**Q1. Protocol and firewall integrity**
- **Clean:** pilot seeds (20268xxx) don't overlap Stage-1 seeds (20269xxx). Anchor, dev and final-test seeds are all distinct. The protocol uses the final checkpoint only, and `final_test_steps=0`. The run and GPU-hour arithmetic checks out (37.3 + 19.1 + 28.2 = 84.6).
- **Retention loophole:** the retention criterion reads "CI lower bound ≥ −1.5% **OR** point ratio ≥ 0.985". The OR clause makes the non-inferiority test optional. Separately, dividing by `|Unif|` is unstable on Minecart and Lunar, where utilities are negative or near zero. The ratio can blow up or flip sign.
- **Multiplicity is underspecified:** the protocol lists 3 primary metrics × 2 inference modes × 3 benchmarks, but Holm is declared only "across benchmarks". It needs one declared primary endpoint per benchmark.
- **Biased regret on approximate references:** `compute_deployed_regret` clips at 0 (`max(0, oracle − deployed)`). Minecart uses an environment reference front and Lunar uses vertex anchors, so the policy can beat the reference. Clipping then hides gains and shrinks the difference between arms.
- **Unbudgeted runs:** the secondary Minecart 2×2 normalization control adds about 20 runs (≈37 GPU-h). These aren't in `stage1a_run_counts`.

**Q2. `simplex_metrics.py`**
- **APM (`magnitude`):** correct. It reduces to Bonin Eq. 2 when U₀>0 and gives APM(F₀;F₀)=0 for any sign. `reference_shifted` correctly refuses non-dominated reference points. One caveat: APM is computed on the max over the deployed return cloud, so it inherits Archive Blindness by construction. Say so in the paper.
- **Vertex cap:** exactly uniform. The cap is a scaled corner simplex, r=εU^{1/(d−1)} gives the right density ∝ r^{d−2}, and the rest of the mass is split by Dirichlet(1).
- **Edge cap: docstring is wrong.** The split is truncated to [0.15, 0.85] and the density omits the (1−m) Jacobian. So it does not sample {wᵢ+wⱼ ≥ 1−ε} uniformly; it samples a truncated, slightly non-uniform mid-edge band. The numerical effect is negligible, but the paper must describe it accurately.
- **`FaceBalancedPreferenceSampler`:** support sizes {2,…,d−1} are correct, there is no outcome input, and the d=2 guard works. No leakage.

**Q3. Novelty and statistics**
- **The registered falsifier was hit.** It reads "Spearman ≥ 0.9" and the Minecart sampled-mode value is 0.90. `claims.json` must explicitly adjudicate this, not just note it as a caveat. Also, ρ on n=5 checkpoints carries almost no inferential weight (ρ=0.8, p≈0.13). Drop any claim that face-cap regret is a *distinct metric*; claim the intervention only.
- **"As d grows" is untestable as designed.** d is fully confounded with environment (d=3 Minecart, d=4 Lunar, d=6 fruit-tree).
- **`claims.json` is stale:** evidence still points to the blueprint and the wording still says the gate is pending. The antecedents also omit prior corner/key-preference training (e.g., PD-MORL); a full-text check is needed.
- **Pilot is sound:** face-cap regret −0.416 (CI [−0.557, −0.253]), interior change +0.018 (CI [−0.029, +0.054]). Pairing is across seeds, not episodes.

**Q4. Reviewer-2 pre-emptions**
1. **"Your d-scaling is an environment comparison."** Cheapest fix: a fruit-tree objective-subset sweep at d ∈ {3,4,5,6}, using the exact oracle on CPU (≈0.013 h/run). Otherwise remove "as d grows" from the claim.
2. **"You trained on the test distribution."** Fixed Face-Balanced puts 25% of its mass exactly on the evaluated vertices; Uniform puts zero there. Cheapest fix: add a static Dirichlet(α<1) sparse-prior arm, at least on fruit-tree (CPU) and Minecart. Make face-cap regret *excluding exact vertices* the primary metric.

---

### GATE-1 VERDICT: **CONDITIONAL PASS**

Blocking fixes, all doable before launch:
1. **Retention criterion:** delete the OR clause and make the CI lower bound ≥ −0.015 the only criterion. On Minecart and Lunar, normalize by (U − wᵀr_ref) instead of `|Unif|`.
2. **One primary endpoint per benchmark:** argmax face-cap mean regret, exact vertices excluded, with the composition and weighting of vertex-cap and edge-cap points stated. Holm runs over the 3 core benchmarks only; everything else is labelled secondary.
3. **Signed regret where there's no exact oracle:** on Minecart and Lunar, report the signed gap as primary plus the fraction of points clipped. Freeze and hash anchor checkpoints (final checkpoint only) before evaluating either arm.
4. **Address the d confound:** add the fruit-tree d-subset sweep and the Dirichlet(α<1) control arm, or reword the claim to drop d-scaling and "beyond trivial sparse priors".
5. **Budget:** either budget the Minecart 2×2 runs (~37 GPU-h) or mark them deferred.
6. **`claims.json`:** adjudicate the sampled ρ=0.90 falsifier, update evidence and allowed wording, and add the corner-preference antecedent. Also fix the `sample_edge_cap_preferences` docstring.

Stage-1A unlocks (`execution_authorized: true`) only after fixes 1–6 are re-verified.

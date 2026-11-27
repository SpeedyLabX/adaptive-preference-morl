# Gate-1B Delta Re-Verification of `stage1-simplex-face-exposure-v1` (v1.1)

- **Reviewer:** Claude Code (**Claude Opus 5.5**, read-only via Windows `claude.exe -p --allowedTools "Read"`)
- **Date:** 2026-11-27
- **Audited Branch / Commit:** `research/week1-gate-step2` @ `aa8bf5b`
- **Audited Scope (Read-Only, 5 Files):**
  1. `research/qualification/gate1_stage1_protocol_audit_v1.md`
  2. `configs/protocols/stage1_simplex_face_exposure_v1.json`
  3. `src/evaluation/simplex_metrics.py`
  4. `research/claims.json`
  5. `research/literature/notes/basaklar2023pdmorl.md`
- **Verdict:** **PASS** (all 6 Gate-1 blocking fixes verified; ready to unlock `execution_authorized: true` upon user GO)

---

## Verbatim Claude Opus 5.5 Gate-1B Response

## Gate-1B delta re-verification of `stage1-simplex-face-exposure-v1` (rev. v1.1)

I read only the 5 files named in the handoff and made no edits.

| # | Fix | Result |
|---|---|---|
| 1 | Retention criterion | **PASS.** The only criterion is now the 95% CI lower bound ≥ −0.015, with no point-estimate fallback. The denominator is the reference-shifted utility U−wᵀr_ref, implemented with a positivity guard in `compute_shifted_interior_retention` (simplex_metrics.py:174). |
| 2 | One primary endpoint per benchmark | **PASS.** The endpoint is argmax non-vertex face-cap mean regret, with exact vertices excluded. The vertex-cap and edge-cap counts and weights are stated for each benchmark; the counts are consistent with 10 points per vertex and 6/4/10/5 per edge. Holm runs over the 3 core benchmarks only, and DST sits outside the family. |
| 3 | Signed regret, clipping, anchor freeze | **PASS.** `mean_signed_gap`, `fraction_clipped` and `cvar_10_signed_gap` are implemented (simplex_metrics.py:151–166). The anchor freeze-and-hash rule covers all 21 anchors and is tied to the final-test firewall. |
| 4 | d confound and sparse-prior control | **PASS.** The protocol adds the fruit-tree d ∈ {3,4,5} sweep (90 CPU runs; the d=6 runs are reused) and a `dirichlet_sparse` mode (α=0.25, range-checked) on fruit-tree and Minecart. |
| 5 | Minecart 2×2 and budget | **PASS.** The 2×2 is deferred and excluded from the budget (20×1.8639 = 37.3 GPU-h). The GPU arithmetic checks out: 30×1.8639 = 55.9, 20×0.9536 = 19.1, anchors 16.8+11.4 = 28.2, total 103.2. Run counts also check out: 146 CPU + 71 GPU = 217. |
| 6 | `claims.json`, PD-MORL note, docstrings | **PASS.** The ρ=0.90 falsifier is adjudicated, metric novelty is dropped, and the claim is now intervention-only. Evidence is updated and PD-MORL is listed as closest antecedent, backed by a page-cited full-text note. The edge-cap docstring is accurate, and the APM Archive Blindness note is present. |

### Remaining flaws (none block the gate)

- **a. The d-scaling test isn't pinned down.** The primary research question and the falsifier both say "growing starvation benefit across d", but neither defines the statistic. Before any checkpoint is evaluated, freeze a specific test, e.g. a paired-seed slope of relative regret reduction against d, its CI, and one-sided p. Otherwise there's room for forking paths.
- **b. CPU hours are understated.** 126×0.0134 + 20×0.02 = 2.09 h, but the protocol says 1.8. CPU runs are also labelled with a `gpu_hours` field. Both are cosmetic.
- **c. The pilot number may use the old endpoint.** The `allowed_wording` in `claims.json` quotes the pilot's "−56.3% face-cap regret". That figure probably used the pre-revision endpoint, which included vertices. Either label it that way or recompute it on the non-vertex endpoint. If it is reported next to Stage-1A numbers as is, it is an apples-to-oranges comparison.
- **d. The PD-MORL note promises 4 differences but lists 3.** Fix the count.
- **e. Leftover field.** `minimum_retention_ratio: 0.985` is still present. It doesn't conflict with the sole criterion, but drop it or label it informational so it can't be read as a fallback.
- **f. Reviewer-2 risk to watch.** Near d=3 vertices the vertex-cap density is roughly linear in distance, so part of the vertex-cap gain may just be interpolation from exact-vertex training. The Dirichlet(0.25) contrast is the defence. Report it prominently, even if it is null.

### GATE-1B VERDICT: **PASS**

All 6 Gate-1 blocking fixes are in place. Stage-1A can be unlocked (`execution_authorized: true`) on the user's GO. The one condition: item (a) must be frozen in the protocol before any Stage-1A checkpoint is evaluated. It does not need to happen before training starts. Items (b)–(e) are housekeeping.

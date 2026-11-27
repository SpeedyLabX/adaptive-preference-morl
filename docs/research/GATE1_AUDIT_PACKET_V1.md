# Gate-1 Pre-Sweep Protocol, Novelty & Scientific Code Audit Packet (v1)

**Target Reviewer:** **Claude Opus 5.5** (`claude`, read-only Gate-1 worker in Orca)  
**Branch / Base Commit:** `research/week1-gate-step2` (Step-1 commit `f221868`, Step-2 commit `ce5066e`)  
**Execution Lock:** `configs/protocols/stage1_simplex_face_exposure_v1.json` has `execution_authorized: false`. **NO Stage-1 sweep may launch until Gate 1 returns PASS (or all blocking findings are resolved and re-verified).**

---

## 1. Exact Read-Only Scope (4 Core Files)

Inspect **ONLY** these 4 files (do not scan the full repository):

1. `configs/protocols/stage1_simplex_face_exposure_v1.json` — Frozen Stage-1 protocol (benchmarks, seeds, splits, 21 vertex anchors, `>= 98.5%` retention margin, adaptive-benefit decision rule, matched-hardware routing).
2. `src/evaluation/simplex_metrics.py` — Implementations of sign-guarded APM (`2609.26918`), Effective Coverage $C$, deployed face-cap regret, Archive Blindness gap, exact $\epsilon$-cap samplers, static `FaceBalancedPreferenceSampler`, and rank-correlation helpers.
3. `research/evidence/week1_fruit_tree_pilot_v1.json` — Week-1 Step-2 10-seed paired pilot on `fruit-tree-v0` ($d=6$, exact discounted oracle; inspect top-level summary/paired statistics, not raw per-seed blobs).
4. `research/claims.json` — Registered `simplex-face-exposure` claim, falsifiers, and novelty verdict (`novelty_unresolved` pending Gate 1).

*(Optional targeted spot-checks only if needed: `research/evidence/week1_minecart_checkpoint_metrics_v1.json` top-level `ranking_analysis` & `apm_negative_utility_sign_guard_witness`, and `src/models/face_exposure_ppo.py`.)*

---

## 2. Summary of Verified Week-1 Inputs

- **Step-1 Full-Text Gate (`f221868`):** Both `arXiv:2605.10585` (*Controllability*) and `arXiv:2609.26918` (*Preference Coverage Collapse*) read in full and registered (`29` papers, `25` read). Neither tests on-policy simplex-face exposure without hindsight relabeling or $d > 3$ boundary starvation.
- **Step-2.1 Minecart 5-Checkpoint Audit (`week1_minecart_checkpoint_metrics_v1.json`):**
  - Negative-utility APM edge case confirmed on `18/71` (`25.4%`) weights: raw Bonin Eq. 2 falsely gives `0.2535` self-APM on the reference front; both `magnitude` and `reference_shifted` guards give `0.0000`.
  - `argmax` rank correlations vs. deployed face-cap regret: Spearman $\rho = 0.80$ ($0.60$ cond-only), APM-Mag $\rho = 0.80$ ($0.60$ cond-only), HV $\rho = -0.40$ (Archive Blindness).
  - *Caveat disclosed:* `sampled` Spearman on $n=5$ is exact `0.90` only because specialist `S` anchors rank 5 ($0.80$ cond-only; APM-Mag is $0.70$ / $0.40$ cond-only).
- **Step-2.2 `fruit-tree-v0` ($d=6$) 10-Seed Pilot (`week1_fruit_tree_pilot_v1.json`):**
  - `argmax` deployed face-cap regret: `0.7383 → 0.3225` (`-56.3%`, $d_z = -1.593$, Holm $p = 0.0195$, 9/10 paired seeds improved; 1 honest negative seed `20268007`).
  - Interior utility retention: `100.40%` (`4.5543 → 4.5724`, 95% CI lower bound `-0.63%`).
  - Power sizing: $n_{80\%}=6$, $n_{90\%}=7$ $\implies$ Stage 1 frozen at `10–12` paired seeds.
- **Live WSL RTX 4060 `hei-research` Throughput Probe (`EXP-810E47201598`, `research/qualification/stage1_throughput_probe_v1.json`):**
  - `minecart-v0` ($d=3$, 4M): `604.4` sps $\to$ `1.864` GPU-h/run ($\le 2.0$h).
  - Full-simplex `mo-lunar-lander-v3` ($d=4$, 4M): `1,219.7` sps $\to$ `0.954` GPU-h/run ($\le 2.0$h).
  - `fruit-tree-v0` ($d=6$, 153.6k): `48.3s` max (`0.0134` h/run on CPU).

---

## 3. Concrete Falsifiable Audit Questions for Claude Opus 5.5

Please answer strictly in **≤ 600 words** with an explicit **GATE-1 VERDICT (`PASS`, `CONDITIONAL PASS` with exact blocking fixes, or `FAIL / PIVOT`)**:

1. **Protocol & Firewall Integrity:** Are there any subtle leakage, seed-overlap, checkpoint-selection, budget-asymmetry, or multiple-testing flaws in `stage1_simplex_face_exposure_v1.json` (including the 21 vertex-specialist anchor runs, `>= 98.5%` interior retention margin, `d=2` DST exclusion rule, and matched-hardware pairing)?
2. **Scientific Code Audit (`src/evaluation/simplex_metrics.py`):** Check the mathematics and edge cases of:
   - `compute_abandoned_preference_mass` (`magnitude` vs. `reference_shifted` sign guards when $U_{\mathcal{F}_0}(\mathbf{w}) \le 0$);
   - `sample_vertex_cap_preferences` (exact uniform measure $1 - w_i = \epsilon U^{1/(d-1)}$) and `sample_edge_cap_preferences`;
   - `FaceBalancedPreferenceSampler` (static exposure weights, zero outcome leakage) and `compute_deployed_regret`.
3. **Novelty & Statistical Validity (`claims.json` + `week1_fruit_tree_pilot_v1.json`):** Given `arXiv:2605.10585` and `arXiv:2609.26918` plus our honest `sampled` Spearman = `0.90` caveat on Minecart ($n=5$), is the Stage-1 claim (`simplex-face-exposure`) cleanly differentiated and appropriately scoped for RLC 2027, or should any claim wording / control arm be tightened before unlocking Stage-1A compute?
4. **Top 2 Reviewer-2 Pre-Emptions:** Name the two sharpest methodological objections an RLC 2027 Reviewer 2 would raise *before* we burn ~85 GPU-hours on Stage-1A, and the cheapest pre-sweep fix for each.

---

## 4. Ready-to-Paste Orca Worker Prompt for Claude Opus 5.5

```text
You are executing Gate 1 (Pre-Sweep Protocol, Novelty & Scientific Code Audit) per CLAUDE.md for SpeedyLabX/adaptive-preference-morl on branch `research/week1-gate-step2`.

Strict rules:
- READ-ONLY. Do NOT edit files, run training, or modify Git state.
- Read ONLY these 4 files (plus docs/research/GATE1_AUDIT_PACKET_V1.md for context):
  1. configs/protocols/stage1_simplex_face_exposure_v1.json
  2. src/evaluation/simplex_metrics.py
  3. research/evidence/week1_fruit_tree_pilot_v1.json (inspect summary/paired_summary only, not full runs array)
  4. research/claims.json
- Hard output cap: <= 600 words.

Answer the 4 falsifiable audit questions in docs/research/GATE1_AUDIT_PACKET_V1.md §3 and end with an explicit GATE-1 VERDICT: PASS, CONDITIONAL PASS (numbered blocking fixes), or FAIL/PIVOT.
```

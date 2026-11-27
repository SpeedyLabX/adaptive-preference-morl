# Q1 / Top-Conference Research Blueprint

**Repository:** `SpeedyLabX/adaptive-preference-morl`
**Synthesized from:** 3-model adversarial review & debate (Antigravity + Claude Code + Codex CLI, 2026-11-27)
**Status:** Strategy locked — pre-execution (requires full-text reading of `arXiv:2605.10585` and `arXiv:2609.26918` before freezing novelty claims)

---

## 1. Target Venues (2-Stage Publication Plan)

1. **Stage 1 — Top Conference: RLC 2027** *(Reinforcement Learning Conference; stretch target: NeurIPS 2027)*
   - Best venue fit for rigorous RL foundations, controlled mechanistic evaluation, and honest negative/static-control results.
2. **Stage 2 — JCR Q1 Journal Extension: IEEE TNNLS** *(IEEE Transactions on Neural Networks and Learning Systems)*
   - JCR Q1, publishes MORL theory + empirical evaluations, generous appendix space for full proofs, ablations, and retained PCQN negative studies.

---

## 2. Core Publishable Thesis & Scientific Boundaries

### Primary Causal Thesis (Stage 1)
> **Simplex-Boundary Collapse in Preference-Conditioned MORL: Theory, Diagnosis, and On-Policy Exposure Fixes**

- Hold the on-policy learner fixed (faithful raw-utility-preserving scalarization, no hindsight relabeling) and test causally whether **simplex-face exposure** reduces held-out **face-cap regret** while meeting a prespecified **interior-utility retention margin**.
- Compare four matched-budget samplers:
  1. **Uniform** (baseline Dirichlet / uniform pool)
  2. **Fixed Face-Balanced** *(mandatory static control — if fixed coverage matches adaptive, publish the static-exposure result honestly)*
  3. **Compute-Matched Random**
  4. **Adaptive Face-Balanced**

### Secondary Mechanistic Control (Clean 2×2 on Minecart)
- Cross **Uniform vs. Fixed Face-Balanced** × **uncompensated vs. raw-utility-preserving scalarization** (keeping PopArt critic machinery and conditioning weights identical).
- Add the **implied-preference `ŵ` diagnostic** comparing achieved return alignment `ŵ` against commanded `w` on **near-face mixed-weight bands**.
  - *Algebraic note:* Positive diagonal rescaling $\operatorname{diag}(\sigma)e_i \propto e_i$ fixes exact simplex vertices; distortion acts on **near-face mixed weights**, not exact corners.
- **Preregistered null rule:** If the normalization main effect and interaction are both null, move normalization distortion to the appendix.

### Claims Explicitly Killed / Scoped Out
- **Kill:** 4-signal adaptive PCQN novelty claim (`research/claims.json` already classifies individual signals as incremental; PCQN failed Minecart — retain PCQN negative results in the appendix).
- **Defer:** Dynamics-shift robustness (separate follow-up paper).
- **Infrastructure only:** Research harness & Orca workflow are reproducibility tooling, not claimed algorithmic contributions.
- **Mandatory pre-freeze reading gate:** Read full text of **`arXiv:2605.10585`** (*Controllability*) and **`arXiv:2609.26918`** (*Preference Coverage Collapse*) before freezing final novelty wording.

---

## 3. Theoretical Contributions (Owned by `@hei1sme`)

1. **Archive-Metric Blindness Theorem (Finite MOMDP Construction):** Construct a finite MOMDP where two policies share identical attainable return sets (identical archive HV and EU), yet one conditioned policy incurs fixed $\Omega(\Delta)$ deployed regret on an $\epsilon$-vertex cap.
2. **Simplex-Boundary Starvation & Query Detection Bound:** Show that under uniform Dirichlet sampling on $\Delta^{d-1}$, probability mass inside an $\epsilon$-cap scales as $O(\epsilon^{d-1})$, derive matched preference-query detection sample bounds, and verify predictions against an exact discounted oracle (`deep-sea-treasure-v0`, and `fruit-tree-v0` $d=6$ in Stage 2).

---

## 4. Experiment Matrix & Verified Compute Budget

### Verified Compute (`hei-research`, 2026-11-27)
- **Local:** NVIDIA RTX 4060 Laptop GPU (`8 GB VRAM`) — pilots & fast CPU/GPU dev
- **Kaggle:** `45 GPU-hours/week` + `20 TPU-hours/week` — primary sweep engine
- **Colab:** `~189 compute units/month` (40 reserve) — overflow

### Stage 1 Conference Matrix (`144` Main Runs)
- **4 MO-Gymnasium Benchmarks:**
  1. `deep-sea-treasure-v0` (exact discounted oracle — theory verification)
  2. `minecart-v0` (3D discrete, sparse delayed credit)
  3. `mo-lunar-lander-v3` (4D discrete — trained across the **full simplex**, fixing the prior single-preference specialist caveat)
  4. `mo-halfcheetah-v5` (continuous control transfer)
- **Runs:**
  - `4 envs × 4 samplers × 8 paired training seeds` = **`128` runs**
  - `+ 16` uncompensated-scalarization Minecart runs (completing the 2×2) = **`144` main runs**
- **Throughput Gate (Week 1):** Timed end-to-end pilots must project ≤ `2 GPU-hours/run` (≤ `288 GPU-hours` total ≈ 6–8 weeks of Kaggle + RTX 4060 + Colab).
- **Evaluation Protocol:**
  - Treat the historical 55-case diagnosis as **exploratory only**.
  - Freeze fresh **vertex, near-face, and interior** preference strata, equal tuning budgets, and MO-Gymnasium default HV reference points before sweeps.
  - Report **deployed face-cap regret** (exact oracle on DST; independent reference policies elsewhere), **HV**, **EU**, **Spearman controllability**, **`ŵ` gap**, and **task success separately** — using **IQM + stratified-bootstrap CIs (`rliable`)** with training seeds as inferential units.

### Stage 2 Journal Extension (IEEE TNNLS)
- Add 6-objective `fruit-tree-v0` ($d=6$ scaling law test) and `mo-hopper-2obj-v5`.
- Add full near-face normalization-distortion theory and replicate the winning exposure intervention on external baselines (**D³PO**, **CAPQL**, **GPI-LS / Envelope**).

---

## 5. Multi-Model Agent Allocation (Inside Orca)

- **Antigravity (`agy` CLI) — ~85–90% (Main Coordinator + Builder + Writer):** Day-to-day planning, coding, `hei-research` recipes, analysis, figures, and LaTeX drafting.
- **Claude Code — ~5–10% (Surgical Gatekeeper — 3–4 calls total):** Respectful use on friend's account, reserved strictly for:
  1. Gate 1: Pre-compute frozen protocol & novelty audit (after reading `2605.10585` & `2609.26918`).
  2. Gate 2: Post-sweep claim & integrity audit against raw JSONs.
  3. Gate 3: Final Reviewer-2 "kill-argument" defense on the LaTeX draft.
- **Codex CLI — ~0–5% (Tactical Reserve):** Hard mathematical proof verification or stubborn PyTorch/CUDA debugging when quota permits.

---

## 6. Realistic Team Task Split (`SpeedyLabX`)

Since Orca, `hei-research`, and the primary GPU/Kaggle/Colab accounts live on `@hei1sme`'s machine, team tasks are split so teammates contribute **high-value PRs and analyses without needing shared credentials**:

| Member | Core Role | Concrete Deliverables (PR-based, No Shared Credentials Needed) |
|---|---|---|
| **`@hei1sme`** (Lead / First Author) | Theory, Orca + `hei-research` compute orchestration, Claude Gatekeeper audits, paper lead | 1. Finite-MOMDP Archive-Blindness proof + starvation/query bounds.<br>2. Run & monitor the 144-run `hei-research` matrix (Local + Kaggle + Colab).<br>3. Lead paper writing & final submission. |
| **`@Am2uocVi3t`** | Environment wrappers, external baselines & code PRs | 1. PR adding `deep-sea-treasure-v0` (with exact discounted oracle) + `mo-halfcheetah-v5` + full-simplex Lunar PPO config (CPU unit-tested).<br>2. Port/verify external baselines (`D³PO`, `GPI-LS`, `CAPQL`) for Stage 2.<br>3. *(Optional)* Run assigned seed slice on own free Kaggle account and PR back compact `result.json`. |
| **`@justHman`** | Literature matrix, evaluation/stats pipeline & figures | 1. Full-text synthesis & comparison table for `arXiv:2605.10585`, `arXiv:2609.26918`, D³PO, CAPQL, GPI-LS, PBMORL, MPFT.<br>2. PR for frozen vertex/near-face/interior splits, corner-cap regret, `ŵ` diagnostic, and `rliable` IQM/CI plotting scripts.<br>3. Publication figures, LaTeX tables & *(optional)* independent Kaggle seed slice. |

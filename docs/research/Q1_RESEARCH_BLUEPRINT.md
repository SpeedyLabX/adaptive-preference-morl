# Q1 / Top-Conference Research Blueprint (Week-1 Gated)

**Repository:** `SpeedyLabX/adaptive-preference-morl`
**Synthesized from:** 3-model adversarial debate + final GO/NO-GO audit (Antigravity + Claude Code + Codex CLI, 2026-11-27)
**Status:** **GO for Week-1 Pilot & Literature Gate; NO-GO for large sweeps until Week-1 STOP/PIVOT gate passes**

---

## 1. Target Venues (2-Stage Publication Plan)

1. **Stage 1 — Top Conference: RLC 2027** *(Reinforcement Learning Conference; stretch: NeurIPS 2027)*
2. **Stage 2 — JCR Q1 Journal Extension: IEEE TNNLS** *(JCR Q1, MORL theory + empirical work + generous appendices)*

---

## 2. Core Publishable Thesis & Registered Claim (`simplex-face-exposure`)

### Primary Hypothesis (Stage 1)
> **Does fixed on-policy simplex-face exposure reduce deployed face-cap regret as objective dimension $d \ge 3$ grows, while preserving interior utility?**

- Hold the on-policy learner fixed (faithful raw-utility-preserving scalarization, **no hindsight relabeling**) and test the primary contrast: **Uniform vs. Fixed Face-Balanced** exposure across $d \in \{3, 4, 6\}$.
- **Staged expansion:** Expand to **Compute-Matched Random** and **Adaptive Face-Balanced** *only if* the Week-1 multi-seed pilot justifies it. Claim adaptive benefit only if Adaptive beats **both** Fixed Face-Balanced and Compute-Matched Random at matched budgets.
- **Secondary 2×2 control (Minecart $d=3$):** Cross Uniform vs. Fixed Face-Balanced × uncompensated vs. raw-utility-preserving scalarization, plus the implied-preference `ŵ` diagnostic on **near-face mixed weights** (positive diagonal PopArt rescaling $\operatorname{diag}(\sigma)e_i \propto e_i$ leaves exact vertices invariant). If null, move to appendix.

### Hard Week-1 STOP/PIVOT Gate (Runs BEFORE Any Large Sweep)
1. **Full-text reading gate:** Read full text of **`arXiv:2605.10585`** (*Controllability*) and **`arXiv:2609.26918`** (*Preference Coverage Collapse*) and register both in `research/literature/`.
2. **Metric & novelty overlap check:** Run a cheap, matched-budget, paired-seed pilot (Uniform vs. Fixed Face-Balanced, relabeling disabled) on **`fruit-tree-v0` ($d=6$, exact oracle)** plus score existing audited Minecart PPO checkpoints with **Spearman (`2605.10585`)**, **APM (`2609.26918`)**, and **deployed face-cap regret**.
3. **Decision rule:**
   - **STOP / PIVOT** if `2609.26918` already covers on-policy face exposure, if Spearman/APM already rank checkpoints nearly identically (Spearman $\gtrsim 0.9$), OR if Fixed Face-Balanced yields no meaningful cap-regret gain within the prespecified interior-utility retention margin.
   - **GO to Stage-1 Sweep** only after freezing the protocol JSON (with exact numeric retention margin & seed count powered from the pilot) and passing **Gate 1**.

### Literature Expansion (Top-Venue Standard)
- Expand the current 28-paper library to **45–60 grounded primary references** across 5 clusters: (1) MORL foundations, (2) 2025–2026 MORL frontier (`2605.10585`, `2609.26918`, D³PO, MORL-Generalization, MORL-FB, PBMORL, MPFT, C-MORL), (3) RL curriculum/UED, (4) PopArt & multi-task gradient conflict (PCGrad/CAGrad/Nash-MTL), and (5) `rliable` statistical evaluation.

---

## 3. Benchmarks ($d \ge 3$ Scaling Suite) & Budgeted Reference Compute

- **Why $d \ge 3$:** At $d=2$, $\epsilon$-cap volume is linear ($O(\epsilon)$), so $d=2$ cannot test exponential $O(\epsilon^{d-1})$ boundary starvation.
- **Core Stage-1 Suite:**
  1. `fruit-tree-v0` ($d=6$, fast CPU, **exact discounted oracle** — tests strong $\epsilon^5$ starvation)
  2. `minecart-v0` ($d=3$, sparse delayed credit)
  3. `mo-lunar-lander-v3` ($d=4$, trained across the **full simplex**)
  4. *Sanity check only:* `deep-sea-treasure-v0` ($d=2$, exact oracle)
- **Reference-Policy Compute (Explicitly Budgeted — No Hidden Cost):**
  - On `fruit-tree-v0` and `deep-sea-treasure-v0`: **exact discounted oracle** (`0` extra training runs).
  - On `minecart-v0` ($d=3$) and `mo-lunar-lander-v3` ($d=4$): primary comparison is **direct paired deployed scalarized utility** across arms; optional vertex-specialist anchors are explicitly budgeted at `7` vertices × `3` seeds = **`21` runs** (owned by `@hei1sme` via `hei-research`).
- **Paired-Seed Power & Hardware Consistency:**
  - Use **10–12 paired training seeds** on the primary Uniform vs. Fixed Face-Balanced contrast (sized from the Week-1 multi-seed pilot) rather than spreading thin across 4 arms prematurely.
  - Keep every paired-seed contrast on **matched hardware/software** (do not mix heterogeneous teammate Kaggle hardware inside a single paired contrast).
- **Stage 2 (IEEE TNNLS Extension):** Add `mo-hopper-v5` ($d=3$ continuous control), full near-face normalization-distortion analysis, and external baseline replication (**D³PO**, **CAPQL**, **GPI-LS / Envelope**).

---

## 4. Multi-Agent Quota-Smart Allocation (Inside Orca)

- **Antigravity (`agy` CLI) — ~65–75%:** Default **`gemini-4-neon-medium`** (`agy --model gemini-4-neon-medium`); escalate via `/model` to `high` for hard proof/protocol/writing turns (`xhigh` only if stuck).
- **Claude Code (`claude`) — Fixed at Claude Opus 5.5 (~20–30% share, read-only default except delegated LaTeX edits):**
  1. **3 Irreversible Scientific Gates:** Gate 1 (Pre-Sweep Protocol, Novelty & Code Audit), Gate 2 (Post-Sweep Claim & Integrity Audit), Gate 3 (Pre-Submission Reviewer-2 Kill-Argument). *Human fallback:* if unavailable, `@hei1sme` + `@Am2uocVi3t` complete Gate 1 manually using `CLAUDE.md` criteria.
  2. **Ongoing scientific work:** Scientific code review (metrics/oracles/PPO), theory proof checks, and LaTeX drafting/polishing (keep prompts targeted to 2–6 explicit files).
- **Codex CLI (`codex`) — Tactical Reserve (`gpt-6.2-sol`, ~0–5%, read-only):** Check quota first; use only for stubborn PyTorch/CUDA bugs or targeted algebraic/proof cross-checks.

---

## 5. Team Task Split (`SpeedyLabX`)

| Member | Core Role | Concrete Deliverables (No Shared Credentials Needed) |
|---|---|---|
| **`@hei1sme`** (Lead / First Author) | Theory, Orca + `hei-research` compute (RTX 4060 / Kaggle / Colab), Gate audits, paper lead | 1. Week-1 full-text gate (`2605.10585` & `2609.26918`) + `fruit-tree-v0` pilot.<br>2. Theory proofs (Archive Blindness + $d$-scaling starvation/query bounds).<br>3. Run matched-hardware paired sweeps & budgeted vertex anchors. |
| **`@Am2uocVi3t`** | Environment wrappers, oracle & baselines (PR-based) | 1. PR adding `fruit-tree-v0` ($d=6$) & `deep-sea-treasure-v0` exact discounted oracles + full-simplex Lunar PPO config (CPU unit-tested).<br>2. Stage-2 external baselines (`D³PO`, `GPI-LS`, `CAPQL`) + `mo-hopper-v5` ($d=3$). |
| **`@justHman`** | Literature expansion, evaluation metrics & figures (PR-based) | 1. Expand literature to 45–60 grounded references + closest-prior-work table.<br>2. PR for fresh vertex/near-face/interior splits, APM + Spearman + deployed cap-regret + `ŵ` diagnostic + `rliable` IQM/CI plots. |

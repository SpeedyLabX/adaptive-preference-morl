---
paper_id: basaklar2023pdmorl
evidence_level: full_text
---

# PD-MORL: Preference-Driven Multi-Objective Reinforcement Learning Algorithm

Primary source: https://openreview.net/forum?id=zS9sRyaPFlJ (arXiv v3: https://arxiv.org/abs/2208.07914)

## Access report

OpenReview full text checked on 2026-10-15; re-verified in full on 2026-11-27 from the 24-page ICLR 2023 camera-ready arXiv v3 PDF (`arXiv:2208.07914v3 [cs.LG]`, 29 May 2023; main text pp. 1–11, Appendices A–B pp. 12–24) stored locally in Git-ignored `research/literature/pdfs/basaklar2023pdmorl.pdf` (SHA-256: `43b55a23a297fd77f142c8ecb83a9ecc2e2f1b2b9eea1cd8c47347647cbc1cde`).

## Relevance and mechanism

- **Off-policy preference-conditioned learner with HER + cosine/angle guidance:** Extends DDQN (`MO-DDQN-HER`, discrete) and TD3 (`MO-TD3-HER`, continuous) with a preference-driven optimality operator / directional angle term, uniform preference sampling across $C_p = 10$ parallel preference sub-spaces, and HER preference relabeling ($N_\omega = 3$ random relabels per transition; Section 4.2, pp. 5–6; Appendix B.1, pp. 16–19).
- **Key-preference corner/center pre-training for preference alignment (Section 4.2, p. 6; Appendix B.1.3 & B.3, pp. 17, 21):** Defines a key preference set comprising the $L$ simplex corners ($\mathbf{e}_1, \dots, \mathbf{e}_L$) plus the simplex center $(1/L, \dots, 1/L)$.
  - On discrete tasks (DST, Fruit Tree Navigation $d=6$), PD-MORL assumes the ground-truth Pareto front is already known and uses oracle corner solutions directly to fit a linear RBF interpolator $I(\boldsymbol{\omega})$ (Appendix B.3, p. 21).
  - On continuous tasks (5 two-objective MO-MuJoCo tasks, $L=2$ only), PD-MORL pre-trains separate fixed-preference MO-TD3 specialist policies for each key preference ($1\text{M}$–$2\text{M}$ steps each, Table 6, p. 21) solely to fit $I(\boldsymbol{\omega})$ for the directional angle loss.

## Evidence and relationship to claims

- **`simplex-face-exposure` (key corner-preference antecedent):** Closest prior work using explicit simplex corner preferences during training. Critically distinct from our Stage-1 intervention on four axes:
  1. PD-MORL uses corners only to fit an auxiliary RBF alignment interpolator (either from oracle discrete fronts or separate specialist pre-training runs), whereas `simplex-face-exposure` tests a **single-stage, zero-oracle, static face-balanced preference sampler** across vertices and proper sparse faces ($s \in \{2, \dots, d-1\}$) inside one shared **on-policy** learner with **no hindsight relabeling**.
  2. PD-MORL relies heavily on off-policy HER relabeling ($N_\omega=3$) and $C_p=10$ parallel sub-space collectors; it does not isolate on-policy simplex-face exposure vs. uniform or static $\mathrm{Dirichlet}(\alpha < 1)$ sparse priors at matched budgets.
  3. Continuous control experiments in PD-MORL are strictly $L=2$; FTN ($d=6$) is evaluated only via front-level Hypervolume and CRF1 (Tables 1 & 7), never deployed face-cap regret excluding exact vertices.
- **`combined-allocation`:** Establishes off-policy conditioned-policy prior art with parallel preference sub-spaces; does not perform a factorial comparison of autonomous curriculum allocation signals.

## Limitations and falsifiers

- Author-acknowledged limitation (Section 5.2, p. 9; Appendix B.4, p. 23): interpolation quality depends heavily on whether pre-trained key corner solutions accurately represent the Pareto front (e.g., `MO-Hopper-v2` corner pre-training bias).
- Evidence pointers: Section 4.2 & Fig. 2 (p. 6); Section 5.1–5.2 & Tables 1–3 (pp. 7–9); Appendix B.1.2–B.1.3 (p. 17); Appendix B.3 Tables 4–6 (p. 21); Appendix B.4 Table 7 (p. 22).

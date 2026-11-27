---
paper_id: bonin2026coverage-collapse
evidence_level: full_text
---

# On Preference Coverage Collapse from Hindsight Relabeling in Multi-Objective Reinforcement Learning

Primary source: https://arxiv.org/abs/2609.26918

## Access report

The 11-page arXiv v1 full text (`arXiv:2609.26918v1 [cs.LG]`, 22 Sep 2026; main text pp. 1–7, references p. 8, Appendices A–E pp. 9–11) was read in full on 2026-11-27 from the local Git-ignored copy `research/literature/pdfs/bonin2026coverage-collapse.pdf` (SHA-256: `867ba725d303a10a6a336de288054bb47cb7beb3a96e176d68216d95e6104e23`). Recorded as a preprint (`source_type: preprint`); PDF remains outside Git.

## Why this work is relevant

This preprint is a mandatory Week-1 STOP/PIVOT gate paper for `simplex-face-exposure` (alongside `delasheras2026controllability`). It provides the first controlled empirical diagnosis of **Preference Coverage Collapse (PCC)** induced by HER-style hindsight preference relabeling in off-policy preference-conditioned MORL, introduces **Effective Coverage ($C$)** and **Abandoned Preference Mass (APM)** as front-level coverage diagnostics, and proposes **`her_mix`** to prevent conditioning-distribution collapse.

## Mechanism and assumptions

### Scope and problem setting

- **Single-policy, off-policy, continuous-action MORL only:** Evaluates four off-policy actor-critic preference-conditioned algorithms spanning a $2 \times 2$ grid over `{update rule} × {preference sampler}`:
  1. **CAPQL** (SAC backbone + angle-cone sampler around the diagonal),
  2. **GPI-PD** (TD3 backbone + GPI weight-support sampler $\mathcal{M}$),
  3. **MO-TD3** (TD3 backbone + CAPQL angle-cone sampler),
  4. **CAPQL-uniform** (SAC backbone + uniform simplex sampler).
- All four algorithms run with library-default hyperparameters; **only the replay buffer varies** (Section 5.1, p. 5; Appendix A, p. 9). No on-policy learners (e.g., PPO/MOPPO) and no non-relabeling simplex-face exposure interventions are tested.

### Relabeling operators and cost-objective clip artifact

- Given a stored transition $(s, a, \mathbf{r}, s', \mathbf{w})$ and achieved objective vector $\hat{\mathbf{o}}$ (per-step reward $\mathbf{r}$, or return-to-go in `her_future`), the standard operator **`her_achieved`** maps $\hat{\mathbf{o}}$ onto the $(d-1)$-simplex via:
  $$\mathbf{w}' = \mathrm{normalize}(\mathrm{clip}(\hat{\mathbf{o}}, 0, \infty)), \qquad \mathrm{normalize}(\mathbf{x}) = \frac{\mathbf{x}}{\sum_i x_i}.$$
- **Hidden clip artifact (Section 2.6, p. 3; Section 6.1, p. 5; Appendix B, p. 9):** For objectives that are negative by construction (control/energy costs on 4 MO-MuJoCo tasks), $\mathrm{clip}(\hat{o}_i, 0, \infty) = 0$ almost everywhere, causing 86–100% of relabels to degenerate to corner or uniform vectors. Replacing clipping with per-objective min-max normalization (**`her_scaled`**) eliminates the clip confound (e.g., `CAPQL/ant-2obj` shifts from $d = -1.41$ to $+0.04$). All main results use `her_scaled`.

### Diagnostic metrics: Effective Coverage ($C$) and Abandoned Preference Mass (APM)

Computed from each run's saved final Pareto front $\mathcal{F}$ on the same discrete evaluation preference grid $W \subset \mathcal{W}$ used for EUM (Section 4.1, p. 4; Section 5.3, p. 5):

1. **Effective Coverage ($C$, Eq. 1, p. 4):** For each $\mathbf{w} \in W$, let $p_k \in \mathcal{F}$ be the maximizing return and $\rho_k$ be the share of $W$ for which $p_k$ is optimal:
   $$C(\mathcal{F}) = \exp\left(-\sum_{k=1}^{|\mathcal{F}|} \rho_k \ln \rho_k\right) \in [1, |\mathcal{F}|].$$
   Purely structural (Hill number / exponential Shannon entropy); cannot separate healthy corner specialization from uniform value loss.
2. **Abandoned Preference Mass ($\mathrm{APM}$, Eq. 2, p. 4):** Value-aware fraction of $W$ where the front's utility falls meaningfully below a baseline front $\mathcal{F}_0$:
   $$\mathrm{APM}(\mathcal{F}; \mathcal{F}_0) = \frac{|\mathcal{A}_\tau|}{|W|}, \qquad \mathcal{A}_\tau = \bigl\{\mathbf{w} \in W : U_{\mathcal{F}}(\mathbf{w}) < (1 - \tau)\, U_{\mathcal{F}_0}(\mathbf{w})\bigr\},$$
   with fixed threshold $\tau = 0.1$ throughout (abandoned when realizing $< 90\%$ of baseline utility). $\mathrm{APM} \in [0, 1]$ and equals $0$ for the baseline itself. Note two structural properties directly relevant to our evaluation protocol:
   - APM is a **functional of the realized Pareto front $\mathcal{F}$** (same front as EUM, p. 4) relative to a **baseline front $\mathcal{F}_0$**, not deployed single-query scalarized regret against an exact oracle.
   - On benchmarks where scalarized utility $U_{\mathcal{F}_0}(\mathbf{w})$ can be negative (e.g., fuel/cost-heavy preferences in `minecart-v0` or `mo-lunar-lander-v3`), the raw multiplicative form $(1 - \tau) U_{\mathcal{F}_0}(\mathbf{w})$ reverses direction unless shifted positive or sign-guarded; this must be documented explicitly when implementing APM for Step-2 checkpoint audits.

### Repair: `her_mix`

- **Convex mixing rule (Section 4.2, Eq. 3, p. 4):** Per transition, blends the achieved preference direction with the originally collected conditioning preference:
  $$\mathbf{w}_{\mathrm{used}} = (1 - \lambda)\,\mathbf{w}_{\mathrm{collected}} + \lambda\,\mathbf{w}_{\mathrm{achieved}},$$
  with a single fixed coefficient $\lambda = 0.25$ across all 36 algorithm-environment settings (no per-setting tuning). Because $\mathbf{w}_{\mathrm{collected}}, \mathbf{w}_{\mathrm{achieved}} \in \mathcal{W}$, $\mathbf{w}_{\mathrm{used}}$ is automatically a valid simplex preference.

## Evidence and experimental protocol

- **Benchmarks and objective dimensions $d$ (Section 5.1, p. 5; Table 6, p. 11):** 9 continuous-action MO-Gymnasium environments — 8 MO-MuJoCo v5 tasks (`hopper-2obj`, `hopper`, `walker2d`, `halfcheetah`, `swimmer`, `ant-2obj`, `ant`, `humanoid`) plus `mountaincar-continuous` — all with **$d \in \{2, 3\}$ objectives** (`mo-reacher-v5` excluded for discrete actions; $d > 3$ explicitly untested). Crossing 4 algorithms × 9 environments yields **36 settings**.
- **Training budget and statistics (Sections 5.2–5.3, p. 5):** 150k environment steps/run (400k horizon check on 2 flat-baseline tasks confirming verdicts persist), evaluation every 15k, **16 paired seeds/setting** (32 on CAPQL hopper tasks; 3 seeds at 60k for mechanism probes). Primary metric: final EUM; secondary: HV (reported only if sign-stable over a $2 \times 2$ reference-point/normalization grid). Cohen's $d$ with 95% BCa paired bootstrap CIs (10k resamples), Holm-Bonferroni correction, TOST equivalence ($\delta_d = 0.5$).
- **Main empirical findings (Sections 6.1–6.6, pp. 5–7; Appendices B–E, pp. 9–11):**
  - **Harm under sign-robust `her_scaled`:** Degrades **19 of 36** settings ($d \le -0.5$, CI excluding 0, down to $d = -4.03$ on `GPI-PD/halfcheetah`), leaves **16** statistically indistinguishable, and improves **1** (`GPI-PD/ant`, $d = +1.03$, EUM $108 \to 124$).
  - **Confound eliminations:** PER alone is inert ($d \in [-0.43, +0.05]$) and PER$\times$HER is unidentifiable on 6/8 envs and opposite-signed on 2/8 (Table 3). Buffer structure (storage, partitioning, eviction, capacity, uniform relabeling; 256 runs at 16 seeds) causes zero harm (Table 5). Early termination amplifies harm ($J = -105.3$ EUM, $p < 0.001$) but does not create it (`swimmer`, `halfcheetah` never terminate early yet reach $d = -2.87, -4.03$). Return-to-go denoising (`her_future`) recovers only 7%; length filtering up to $L_{\min}=200$ recovers at most 61% (Table 4). Relabeled TD error is *smaller* than collected-conditioning TD error in 34/36 settings, and TD-gated relabeling fails its criterion (62% recovery on `halfcheetah`/`swimmer`).
  - **Coverage collapse and APM tracking:** Effective coverage $C$ drops on 22/32 settings with usable fronts (mean $2.6 \to 1.9$); mean APM on harmed settings is **69%**. APM tracks Cohen's $d$ harm tightly (**Spearman $\rho = -0.73, p < 10^{-5}$**) vs. weak correlation for $C$ ($\rho = +0.40$). Probabilistic relabeling shows a sharp **cliff** rather than a slope: performance holds for $f \le 0.5$ and collapses only as $f \to 1$ ($d = -3.48$ at $f=1$, Figure 4a).
  - **`her_mix` ($\lambda = 0.25$) recovery:** Satisfies pre-registered criteria on **32 of 36** settings (465 additional runs, Table 6): recovers **16/19 harmed settings** (the 3 exceptions have small $|d_{\mathrm{HER}}| \approx 0.56$–$0.63$), reduces mean APM from **69% to 6%** and restores 87% of lost $C$, amplifies the 1 helped setting (`GPI-PD/ant`: $d = +1.03 \to +2.42$, and $+4.46$ at $\lambda = 0.5$), and leaves 15/16 neutral settings unchanged (1 regression: `GPI-PD/mountaincar-continuous`, $d = -0.66$). Recovery stays flat at 94–103% for $\lambda \le 0.5$ on `hopper-2obj`, beating the $(1-\lambda)$ pure-attenuation line (Figure 4b).

## Relationship to project claims

- **`simplex-face-exposure` (Week-1 Step-1 Full-Text Overlap Verdict: CLEAR PASS — NO STOP/PIVOT TRIGGERED):**
  - **Orthogonal failure mechanism and learner family:** Bonin et al. isolate off-policy replay-buffer hindsight relabeling collapse (overwriting collected preferences with achieved directions in SAC/TD3 buffers). They do **not** test on-policy learners without hindsight relabeling (e.g., PPO), do **not** test fixed simplex-face / boundary exposure vs. uniform sampling, and do **not** study geometric $O(\epsilon^{d-1})$ boundary starvation as $d$ grows beyond 3 ($d \in \{2, 3\}$ only; $d > 3$ and discrete-action environments are explicitly listed as untested in Section 7, p. 7).
  - **Distinct diagnostic metric:** APM is a front-level, baseline-relative, uniformly weighted, binary-thresholded ($\tau = 0.1$) fraction over $W$. It does not separate deployed single-query regret from archive-front utility, does not anchor against exact discounted oracles, and weights $\epsilon$-caps around simplex faces only by their shrinking uniform measure $O(\epsilon^{d-1})$. Whether APM ranks existing audited Minecart ($d=3$) PPO checkpoints similarly to deployed face-cap regret (threshold: Spearman $< 0.9$) remains to be verified empirically in Week-1 Step 2.
  - **Novelty status:** Keep `simplex-face-exposure` at `novelty_unresolved` until Week-1 Step-2 empirical overlap check + `fruit-tree-v0` ($d=6$) pilot and Gate-1 audit complete.
- **`combined-allocation`:** Provides strong independent justification for disabling hindsight relabeling in the Stage-1 on-policy contrast and supplies APM as an external diagnostic comparator.
- **Literature expansion lead:** Section 3.4 (p. 4) cites Kubo, Nakanishi, & Ishii (2026), *A Single Deep Preference-Conditioned Policy for Learning Pareto Coverage Sets* (`arXiv:2605.08946`, STCH + CMDPI), a strong candidate for our Stage-1 45–60 reference expansion.

## Limitations and possible falsifiers

- Explicitly bounded by the authors (Section 7, p. 7) to continuous MO-MuJoCo / `mountaincar-continuous` with $d \in \{2, 3\}$ and four single-policy off-policy actor-critic configurations; discrete-action domains, $d > 3$, on-policy learners, multi-policy algorithms, and model-based methods are untested.
- `her_mix` at $\lambda = 0.25$ is a default rather than a universal guarantee (1 regression on `GPI-PD/mountaincar-continuous`; partial recovery on `GPI-PD/halfcheetah`).
- APM depends on the quality and sign convention of the reference baseline front $\mathcal{F}_0$; if $\mathcal{F}_0$ itself suffers uniform-sampling face starvation, APM cannot detect shared boundary suboptimality.

## Evidence pointers

- Section 1 & Figure 1 (pp. 1–2): Motivation and illustration of Preference Coverage Collapse on `hopper`.
- Section 2.6 & Section 6.1 (pp. 3, 5) + Appendix B, Table 2 (p. 9): `her_achieved` clip artifact on negative cost objectives vs. sign-robust `her_scaled`.
- Section 4.1, Eqs. 1–2 (p. 4): Exact definitions of Effective Coverage $C(\mathcal{F})$ and Abandoned Preference Mass $\mathrm{APM}(\mathcal{F}; \mathcal{F}_0)$ ($\tau = 0.1$).
- Section 4.2, Eq. 3 (p. 4): Exact `her_mix` convex combination ($\lambda = 0.25$).
- Sections 5.1–5.3 (pp. 4–5) + Appendix A (p. 9): 4 off-policy algorithms, 9 MO-Gymnasium continuous tasks ($d \in \{2, 3\}$), 150k steps, 16 paired seeds, statistical tests.
- Sections 6.2–6.6, Figures 2–4, Table 1 (pp. 5–7) + Appendices C–E, Tables 3–6 (pp. 9–11): Full 36-setting results, confound eliminations, dose-response cliff, and per-setting `her_mix` validation.
- Section 7 (p. 7): Explicit limitations confirming discrete-action tasks, $d > 3$, and non-off-policy/multi-policy settings remain untested.

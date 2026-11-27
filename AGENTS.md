# Agent Bootstrap — SpeedyLabX / adaptive-preference-morl

This file is the canonical entry point for **Antigravity (`agy` CLI)**, **Codex CLI**, **Claude Code**, and any other agent working in this repository (including inside **Orca**).

---

## 1. Session Start

Before editing any file or launching compute:

1. Run:
   ```bash
   python3 scripts/research_harness.py doctor
   python3 scripts/research_harness.py status
   python3 scripts/research_harness.py roadmap list
   ```
2. Read `docs/research/Q1_RESEARCH_BLUEPRINT.md` — this is the **locked 2-stage research strategy** (Stage 1: **RLC 2027** conference paper; Stage 2: **IEEE TNNLS** JCR Q1 journal extension).
3. Read `docs/research/PROJECT_STATUS.md`, `docs/research/ROADMAP.md`, and `README.md` for the latest verified empirical state and active phase gate.
4. For codebase/architecture questions, **query `graphify-out/graph.json` first** (when present locally) before brute-force grepping across files.
5. If changing protocols, metrics, or claims, also read `docs/ARCHITECTURE.md`, `research/claims.json`, and the relevant JSON in `configs/protocols/`.

---

## 2. Orca Multi-Agent Fleet & Model Routing Policy

This repository is operated from **Orca** with three agent harnesses. Follow this routing policy strictly:

| Agent | Role & Share | Quota-Smart Model & Effort (Late-2026) | When to Use | Hard Rules |
|---|---|---|---|---|
| **Antigravity (`agy` CLI)** | **Main Coordinator, Builder & Writer** (~65–75%) | **Default: Gemini 4 Neon (`medium`)** (`agy --model gemini-4-neon-medium`) for routine turns; bump to **`high`** for hard proof/protocol/writing turns (use **`xhigh`** only if genuinely stuck) | Day-to-day orchestration, literature expansion, code PRs, `hei-research` compute dispatch, statistical analysis, figures, LaTeX drafting | Owns repo state, `research_harness.py` updates, and worker delegation. Never burn `xhigh` on whole sessions or routine work. |
| **Claude Code (`claude`)** | **Cross-Family Gatekeeper, Scientific Reviewer & Writing Partner** (~20–30%) | **Claude Opus 5.5** (fixed; default read-only unless editing LaTeX/text when explicitly requested) | 1. **3 irreversible scientific gates:** Gate 1 (Pre-Sweep Protocol & Novelty), Gate 2 (Post-Sweep Claim & Integrity), Gate 3 (Adversarial Reviewer-2)<br>2. **Scientific code review** (metrics, oracles, PPO/allocation logic)<br>3. **Theory proof checks** and **LaTeX drafting/polishing** | Fixed at Opus 5.5; keep scope explicit (2–6 targeted files) rather than full-repo exploration. Do not waste on trivial formatting or boilerplate debugging. |
| **Codex CLI (`codex`)** | **Tactical Reserve** (~0–5%) | **GPT-6.2 Sol (`gpt-6.2-sol`)** (token-efficient default, read-only) | Stubborn PyTorch/CUDA/numerical bugs or targeted algebraic/proof cross-checks | **Constrained quota:** Check quota/availability before invoking; never use for routine tasks. |

### Worker Handoff Contract
- Whenever `agy` delegates to **Claude Code** or **Codex CLI**, pass:
  1. The exact read-only scope (explicit 2–4 files to inspect, never "read the whole repo").
  2. The concrete falsifiable question or audit gate.
  3. A hard output length cap (~400–650 words) unless a full line-by-line proof check is required.

---

## 3. Compute Orchestration (`hei-research` & WSL Boundary)

- Keep active repository checkouts and Python/Conda environments in **WSL Ubuntu**.
- Multi-backend GPU sweeps (Local RTX 4060 + Kaggle GPU + Google Colab) are managed by `hei-research` (`$HOME/.local/bin/hei-research`).
- From a **Windows Orca agent terminal**, invoke `hei-research` through WSL using absolute Linux paths:
  ```powershell
  wsl.exe -d Ubuntu-24.04 -- bash -lc '$HOME/.local/bin/hei-research --json doctor'
  wsl.exe -d Ubuntu-24.04 -- bash -lc '$HOME/.local/bin/hei-research --json quota'
  ```
- **Always check `--json doctor` and `--json quota` before launching sweeps.**
- Run Week-1 timed end-to-end pilots (including evaluation overhead) to verify ≤ `2 GPU-hours/run` before committing to the full 144-run conference matrix.
- Never commit personal usernames, Windows Desktop/OneDrive paths, credentials, tokens, or checkpoints to Git.

---

## 4. Sources of Truth (Precedence)

When sources disagree, follow this order:

1. Executed tests and versioned result artifacts (`research/evidence/`, `research/qualification/`).
2. Code and frozen protocol/config files (`src/`, `scripts/`, `configs/`).
3. Machine-readable registries (`research/roadmap.json`, `research/claims.json`, `research/experiments.json`) plus `docs/research/Q1_RESEARCH_BLUEPRINT.md`, `docs/research/PROJECT_STATUS.md`, and `docs/ARCHITECTURE.md`.
4. Other Markdown documents.
5. Historical course proposal documents (immutable archival record only).

---

## 5. Scientific Guardrails (Non-Negotiable)

- **Mandatory full-text literature gate:** Never freeze novelty claims until full texts of **`arXiv:2605.10585`** (*Controllability*) and **`arXiv:2609.26918`** (*Preference Coverage Collapse*) are read and registered. Never infer a completed reading from metadata or an abstract.
- **No hype or premature claims:** Never describe a planned, exploratory, or smoke-tested result as experimentally supported. Treat the historical 55-case diagnosis as **exploratory only**.
- **Final-test firewall:** Never tune normalization, hypervolume reference points, checkpoints, or hyperparameters on the final test split.
- **Budget separation:** Keep training, update, curriculum-evaluation, and final-evaluation budgets strictly separate and matched across arms.
- **Mandatory static controls:** Compare adaptive allocation against **Uniform**, **Fixed Face-Balanced**, and **Compute-Matched Random** using paired training seeds. Claim adaptive benefit **only** if it beats both Fixed Face-Balanced and Compute-Matched Random at matched budgets; if Fixed Face-Balanced ties Adaptive, report the static-exposure result honestly.
- **Exact-oracle & metric separation:** Measure deployed scalarized regret against exact discounted oracles (`deep-sea-treasure-v0`, `fruit-tree-v0`) or independent reference policies elsewhere; report task success separately from scalarized utility.
- **Algebraic accuracy on PopArt:** Positive diagonal rescaling $\operatorname{diag}(\sigma)e_i \propto e_i$ leaves exact simplex vertices invariant; normalization distortion applies to **near-face mixed-weight regions**, never exact corners.
- **Provenance:** Preserve raw and discounted vector returns, seeds, protocol/checkpoint SHA-256 hashes, and dependency provenance for every reported result.

---

## 6. Engineering Guardrails

- Target **Python 3.11** with pinned dependencies in `requirements.txt`.
- Create a dedicated branch / Orca worktree for major changes; preserve unrelated user modifications.
- Keep checkpoints (`artifacts/`), generated packages (`outputs/`), logs, caches, paper PDFs, and `graphify-out/` out of Git.
- After code/config changes, run:
  ```bash
  python -m ruff format --check .
  python -m ruff check .
  python -m mypy src scripts
  python -m pytest -q
  python -m bandit -q -r src scripts -ll
  python -m compileall -q src scripts tests
  git diff --check
  ```
- Keep `research/claims.json`, `research/experiments.json`, `research/roadmap.json`, and `docs/research/PROJECT_STATUS.md` synchronized through `scripts/research_harness.py`.

# Orca Kickoff Guide & Model Selection (`agy` CLI)

Use this guide when opening your local WSL checkout (`adaptive-preference-morl`) in **Orca**.

---

## 1. Which Model & Effort to Initialize `agy` CLI With (Late-2026 Stack)

Choose your startup model by phase:

| Phase / Worker | Recommended Model & Startup Command | Why |
|---|---|---|
| **Start `agy` CLI (Kickoff, Literature Gate, Theory/Proofs, Protocol Freeze, Paper Writing)** *(Start here!)* | **Gemini 4 Neon (Xhigh):** `agy --model gemini-4-neon-xhigh` | Strongest Antigravity reasoning tier for reading full papers, proving the Archive-Blindness & Starvation theorems, catching protocol flaws, and drafting RLC 2027 / IEEE TNNLS prose. |
| **Switch `agy` CLI (Routine Implementation, Unit Tests, `hei-research` Recipes, Sweep Monitoring)** | **Gemini 4 Neon (Medium):** `agy --model gemini-4-neon-medium` (or `--effort medium`) | Fast, cost-efficient for mechanical coding, pytest fixes, plotting scripts, and monitoring Kaggle/Colab jobs. |
| **Worker — Claude Code (`claude`)** | **Claude Opus 5.5** (or **Claude Fable 5.2** for deepest audits; **Sonnet 5.5** for quick checks), read-only | Use **ONLY** at Gates 1, 2, and 3 (3–4 surgical calls total; respectful of shared account). |
| **Worker — Codex CLI (`codex`)** | **GPT-6.2 Astra** (deep math/proof verification) or **GPT-6.2 Sol (`gpt-6.2-sol`)** (token-efficient PyTorch/CUDA debugging), read-only | Tactical reserve ONLY (check quota first). |

Startup command in Orca's coordinator terminal:
```bash
agy --model gemini-4-neon-xhigh
```

---

## 2. Copy-Paste Kickoff Prompt for `agy` CLI in Orca

```text
You are the Main Research Coordinator for SpeedyLabX/adaptive-preference-morl targeting RLC 2027 (conference) and IEEE TNNLS (JCR Q1 journal extension).

First, complete your mandatory bootstrap:
1. Read AGENTS.md and docs/research/Q1_RESEARCH_BLUEPRINT.md.
2. Run `python3 scripts/research_harness.py doctor`, `status`, and `roadmap list`.
3. Check live compute readiness via `$HOME/.local/bin/hei-research --json doctor` and `--json quota` (through WSL if running from a Windows terminal).
4. Verify that `graphify-out/graph.json` and the symlinked audited PPO checkpoints under `artifacts/` resolve and are readable in this checkout/worktree.

Acknowledge your multi-agent routing policy from AGENTS.md:
- You (`agy`, Gemini 4 Neon) own ~85-90% of work (`gemini-4-neon-xhigh` for theory/planning/writing; `gemini-4-neon-medium` for routine coding/tests/sweeps).
- Claude Code (`claude`, Opus 5.5 / Fable 5.2) is reserved STRICTLY for our 3 surgical read-only Gatekeeper audits (Gate 1 Pre-Sweep Protocol Audit, Gate 2 Post-Sweep Claim Audit, Gate 3 Reviewer-2 Kill-Argument) — do NOT invoke it for routine tasks.
- Codex CLI (`codex`, GPT-6.2 Astra / GPT-6.2 Sol) is tactical reserve only (check quota first; use only for hard proof/algebra checks or stubborn CUDA/PyTorch bugs).

Once bootstrapped, present a crisp Week-1 execution plan following this strict order:
1. Full-text reading and registration of `arXiv:2605.10585` (Controllability) and `arXiv:2609.26918` (Preference Coverage Collapse) to lock the exact novelty boundary.
2. Exact-oracle setup for `deep-sea-treasure-v0` + approved timed 1-seed throughput pilots on non-final preferences (Minecart, full-simplex Lunar, `mo-halfcheetah-v5`) to verify the ≤2 GPU-hr/run budget gate and confirm whether each env has a default MO-Gymnasium HV reference point (otherwise calibrate pilot-only).
3. Freeze the Stage-1 144-run protocol JSON: all four arms (Uniform, Fixed Face-Balanced, Compute-Matched Random, Adaptive Face-Balanced), 8 paired seeds, separately matched budgets, fresh vertex / near-face / interior evaluation strata, explicit interior-utility retention margin, and the decision rule claiming adaptive benefit ONLY if Adaptive beats both Fixed Face-Balanced and Compute-Matched Random.
4. Run Claude Gate 1 (Pre-Sweep Protocol Audit) BEFORE launching the 144-run sweep.

Wait for my confirmation before modifying code or launching any compute.
```

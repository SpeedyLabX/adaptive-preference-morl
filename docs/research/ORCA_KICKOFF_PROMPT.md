# Orca Kickoff Guide & Quota-Smart Model Selection (`agy` CLI)

Use this guide when opening your local WSL checkout (`adaptive-preference-morl`) in **Orca**.

---

## 1. Quota-Smart Model & Effort Policy (Late-2026)

Do **not** burn `xhigh` or heavy models across whole sessions. Default to **`medium`** and escalate only for specific hard turns:

| CLI / Role | Exact Model & Command | When to Use |
|---|---|---|
| **`agy` CLI — Default (~85% of work)** *(Start here!)* | **Gemini 4 Neon (Medium):** `agy --model gemini-4-neon-medium` | Kickoff, bootstrap, literature search/registry, coding, unit tests, `hei-research` recipes, sweep monitoring, plotting, and initial paper drafts. |
| **`agy` CLI — Targeted Escalation (~15% of turns)** | Switch via `/model` to **Gemini 4 Neon (`high`)** *(use `xhigh` ONLY if genuinely stuck on a proof)* | Single high-stakes turns: verifying the Archive-Blindness & Starvation proofs, locking the 144-run protocol, or polishing the final paper abstract/intro. Switch right back to `medium` afterward. |
| **Worker — Claude Code (`claude`)** | **Claude Opus 5.5** (fixed, read-only) | **ONLY at Gates 1, 2, and 3** (3 short surgical calls total; respectful of shared account). |
| **Worker — Codex CLI (`codex`)** | **GPT-6.2 Sol (`gpt-6.2-sol`)** (read-only) | Tactical reserve ONLY (check quota first) — stubborn PyTorch/CUDA bugs or targeted algebra/proof checks. |

Startup command in Orca's coordinator terminal:
```bash
agy --model gemini-4-neon-medium
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

Acknowledge your quota-smart multi-agent routing policy from AGENTS.md:
- You (`agy`, Gemini 4 Neon) stay on `gemini-4-neon-medium` by default for ~85% of work, escalating to `high` only for isolated proof/protocol/writing turns (never burning `xhigh` across whole sessions).
- Claude Code (`claude`) is fixed at **Claude Opus 5.5** and reserved STRICTLY for our 3 surgical read-only Gatekeeper audits (Gate 1 Pre-Sweep Protocol Audit, Gate 2 Post-Sweep Claim Audit, Gate 3 Reviewer-2 Kill-Argument).
- Codex CLI (`codex`, `gpt-6.2-sol`) is tactical reserve only (check quota first).

Once bootstrapped, present a crisp Week-1 execution plan following this strict order:
1. Literature expansion (target 45-60 grounded references across MORL foundations, 2025-2026 frontier, RL curriculum/UED, multi-task gradient conflict/PopArt, and `rliable` evaluation), starting with mandatory full-text reading and registration of `arXiv:2605.10585` (Controllability) and `arXiv:2609.26918` (Preference Coverage Collapse) to lock our exact novelty boundary.
2. Exact-oracle setup for `deep-sea-treasure-v0` + approved timed 1-seed throughput pilots on non-final preferences (Minecart, full-simplex Lunar, `mo-halfcheetah-v5`) to verify the ≤2 GPU-hr/run budget gate and confirm whether each env has a default MO-Gymnasium HV reference point (otherwise calibrate pilot-only).
3. Freeze the Stage-1 144-run protocol JSON: all four arms (Uniform, Fixed Face-Balanced, Compute-Matched Random, Adaptive Face-Balanced), 8 paired seeds, separately matched budgets, fresh vertex / near-face / interior evaluation strata, explicit interior-utility retention margin, and the decision rule claiming adaptive benefit ONLY if Adaptive beats both Fixed Face-Balanced and Compute-Matched Random.
4. Run Claude Gate 1 (Opus 5.5, Pre-Sweep Protocol Audit) BEFORE launching the 144-run sweep.

Wait for my confirmation before modifying code or launching any compute.
```

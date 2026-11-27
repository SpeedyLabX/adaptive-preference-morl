# Orca Kickoff Guide & Quota-Smart Model Selection (`agy` CLI)

Use this guide when opening your local WSL checkout (`adaptive-preference-morl`) in **Orca**.

---

## 1. Quota-Smart Model & Effort Policy (Late-2026)

| CLI / Role | Exact Model & Command | When to Use |
|---|---|---|
| **`agy` CLI — Default (~85% of work)** *(Start here!)* | **Gemini 4 Neon (Medium):** `agy --model gemini-4-neon-medium` | Kickoff, bootstrap, literature expansion, coding, unit tests, `hei-research` recipes, sweep monitoring, plotting, and first drafts. |
| **`agy` CLI — Targeted Turns (~15%)** | Switch via `/model` to **Gemini 4 Neon (`high`)** *(use `xhigh` ONLY if stuck)* | Single hard turns: proofs, protocol freeze, abstract/intro polish. Switch back to `medium` afterward. |
| **Worker — Claude Code (`claude`)** | **Claude Opus 5.5** (fixed, read-only) | **ONLY 3 short surgical calls total** (Gates 1, 2, 3; human checklist fallback if courtesy account is unavailable). |
| **Worker — Codex CLI (`codex`)** | **GPT-6.2 Sol (`gpt-6.2-sol`)** (read-only) | Tactical reserve ONLY (check quota first). |

Startup command in Orca's coordinator terminal:
```bash
agy --model gemini-4-neon-medium
```

---

## 2. Copy-Paste Kickoff Prompt for `agy` CLI in Orca

```text
You are the Main Research Coordinator for SpeedyLabX/adaptive-preference-morl targeting RLC 2027 (conference) and IEEE TNNLS (JCR Q1 journal extension).

First, complete your mandatory bootstrap:
1. Read AGENTS.md, docs/research/Q1_RESEARCH_BLUEPRINT.md, and research/claims.json (note the registered `simplex-face-exposure` hypothesis).
2. Run `python3 scripts/research_harness.py doctor`, `status`, and `roadmap list`.
3. Check live compute readiness via `$HOME/.local/bin/hei-research --json doctor` and `--json quota` (through WSL if running from a Windows terminal).
4. Verify that `graphify-out/graph.json` and the symlinked audited PPO checkpoints under `artifacts/` resolve and are readable in this checkout/worktree.

Acknowledge your quota-smart routing policy from AGENTS.md:
- Stay on `gemini-4-neon-medium` by default (~85% of work); escalate to `high` only for isolated proof/protocol/writing turns.
- Claude Code is fixed at **Claude Opus 5.5** and reserved STRICTLY for 3 surgical read-only Gatekeeper audits (Gates 1, 2, 3 — with human fallback if unavailable).
- Codex CLI (`gpt-6.2-sol`) is tactical reserve only (check quota first).

We are **GO for Week-1 Pilot & Literature Gate, NO-GO for large sweeps until the Week-1 STOP/PIVOT gate passes**. Present a crisp Week-1 plan in this exact order:
1. Full-text reading and registration of `arXiv:2605.10585` (Controllability) and `arXiv:2609.26918` (Preference Coverage Collapse), plus kick off the 5-cluster expansion toward 45-60 grounded references.
2. Run the cheap Week-1 STOP/PIVOT pilot (relabeling disabled, Uniform vs. Fixed Face-Balanced) on `fruit-tree-v0` (d=6, exact discounted oracle) and score existing audited Minecart (d=3) checkpoints with Spearman (2605.10585), APM (2609.26918), and deployed face-cap regret. Report whether our metric adds distinct signal (Spearman < 0.9) and whether Fixed Face-Balanced improves deployed cap-regret within an interior-utility margin.
3. ONLY if Week-1 passes: size paired seeds (10-12 seeds on matched hardware) for the d>=3 core suite (`fruit-tree-v0` d=6, `minecart-v0` d=3, full-simplex `mo-lunar-lander-v3` d=4; DST d=2 as sanity check only), explicitly budget any vertex-specialist anchors, freeze the protocol JSON (with numeric interior retention margin and adaptive-benefit decision rule), and run Gate 1 (Claude Opus 5.5) BEFORE launching any large sweep.

Wait for my confirmation before modifying code or launching any compute.
```

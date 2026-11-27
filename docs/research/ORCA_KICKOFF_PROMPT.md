# Orca Kickoff Guide & Model Selection (`agy` CLI)

Use this guide when opening your local WSL checkout (`adaptive-preference-morl`) in **Orca**.

---

## 1. Which Model to Initialize `agy` CLI With

Choose your `agy` CLI startup model/effort by phase:

| Phase | Recommended `agy` Model & Effort | Why |
|---|---|---|
| **Kickoff, Literature Gate, Theory/Proofs, Protocol Freeze, Paper Writing** *(Start here!)* | **Gemini Pro** (highest available reasoning model) with `--effort high` | Best for reading full papers, proving the Archive-Blindness & Starvation theorems, catching protocol flaws, and drafting RLC / TNNLS prose. |
| **Routine Implementation, Unit Tests, `hei-research` Recipes, Sweep Monitoring** | **Gemini Flash** (or default Medium model) with `--effort medium` | Faster and cheaper for mechanical coding, pytest fixes, plotting scripts, and checking Kaggle/Colab job status. |
| **Worker — Claude Code (`claude`)** | **Claude Sonnet / Opus** (read-only) | Use **ONLY** at Gates 1, 2, and 3 (3–4 surgical calls total; respectful of shared account). |
| **Worker — Codex CLI (`codex`)** | **Codex / GPT reasoning model** (read-only) | Tactical reserve ONLY (when quota allows) for hard proof checks or stubborn PyTorch/CUDA bugs. |

Example startup command in Orca's coordinator terminal:
```bash
agy --effort high
```
*(Or select your highest Gemini Pro reasoning model in Orca's agent model picker before pasting the kickoff prompt below.)*

---

## 2. Copy-Paste Kickoff Prompt for `agy` CLI in Orca

```text
You are the Main Research Coordinator for SpeedyLabX/adaptive-preference-morl targeting RLC 2027 (conference) and IEEE TNNLS (JCR Q1 journal extension).

First, complete your mandatory bootstrap:
1. Read AGENTS.md and docs/research/Q1_RESEARCH_BLUEPRINT.md.
2. Run `python3 scripts/research_harness.py doctor`, `status`, and `roadmap list`.
3. Check live compute readiness via `$HOME/.local/bin/hei-research --json doctor` and `--json quota` (through WSL if running from a Windows terminal).
4. Note that `graphify-out/graph.json` is available locally for fast codebase queries, and audited PPO baseline checkpoints are symlinked under `artifacts/`.

Acknowledge your multi-agent routing policy from AGENTS.md:
- You (`agy`) own ~85-90% of work (planning, coding, theory drafts, `hei-research` sweeps, analysis, LaTeX).
- Claude Code (`claude`) is reserved STRICTLY for our 3 surgical read-only Gatekeeper audits (Gate 1 Pre-Compute Protocol Audit, Gate 2 Post-Sweep Claim Audit, Gate 3 Reviewer-2 Kill-Argument) — do NOT invoke it for routine tasks.
- Codex CLI (`codex`) is tactical reserve only (check quota first; use only for hard proof checks or stubborn CUDA/PyTorch bugs).

Once bootstrapped, present a crisp Week-1 execution plan covering ONLY our first 3 unblocking milestones:
1. Full-text reading and registration of `arXiv:2605.10585` (Controllability) and `arXiv:2609.26918` (Preference Coverage Collapse) so we can lock the exact novelty boundary.
2. Exact-oracle setup for `deep-sea-treasure-v0` + timed 1-seed throughput pilots (Minecart, full-simplex Lunar, `mo-halfcheetah-v5`) to verify the ≤2 GPU-hr/run budget gate.
3. Freezing the Stage-1 144-run protocol JSON (with fresh vertex / near-face / interior evaluation strata and Fixed Face-Balanced static control) ready for Claude Gate 1 review.

Wait for my confirmation before modifying code or launching any compute.
```

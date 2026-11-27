#!/usr/bin/env python3
"""Sequential/resumable hei-research dispatcher for Stage-1A GPU anchors and arms.

Enforces the frozen Gate-1B Stage-1A execution order:
  1. --phase anchors : runs all 21 vertex-specialist anchor recipes (12 Lunar + 9 Minecart,
     or filtered by --env) via hei-research on a single matched backend (default: local RTX 4060),
     skipping runs whose verified 4M-step checkpoint & receipt already exist, and freezes the
     cryptographic anchor manifest when all 21 anchors complete.
  2. --phase arms    : refuses to start unless research/evidence/stage1a_vertex_anchors_frozen_v1.json
     exists and verifies cleanly; then runs paired seeds (all arms for seed s back-to-back on
     identical hardware/backend) via hei-research.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "protocols" / "stage1_simplex_face_exposure_v1.json"
RECIPE_ROOT = REPO_ROOT / "configs" / "recipes" / "stage1a"
ANCHOR_MANIFEST = REPO_ROOT / "research" / "evidence" / "stage1a_vertex_anchors_frozen_v1.json"
HEI_BIN = Path(os.environ.get("HEI_RESEARCH_BIN", str(Path.home() / ".local" / "bin" / "hei-research")))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_anchor_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Frozen anchor manifest missing ({path}). Complete and freeze all 21 vertex anchors first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    recorded = data.get("manifest_sha256")
    body = {k: v for k, v in data.items() if k != "manifest_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    if recorded != actual or int(data.get("total_anchors", 0)) != 21 or int(data.get("final_test_steps", -1)) != 0:
        raise RuntimeError(f"Frozen anchor manifest failed integrity check: {path}")
    return data


def _is_run_complete(kind: str, env_id: str, tag: str, expected_steps: int) -> bool:
    base = REPO_ROOT / "artifacts" / "stage1a-gpu" / ("anchors" if kind == "anchor" else "arms") / env_id
    ckpt = base / f"{tag}.pt"
    receipt = base / f"{tag}_receipt.json"
    if not ckpt.exists() or not receipt.exists():
        return False
    try:
        rec = json.loads(receipt.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        int(rec.get("training_interactions", -1)) == expected_steps
        and int(rec.get("final_test_steps", -1)) == 0
        and _file_sha256(ckpt) == rec.get("checkpoint_sha256")
    )


def _sync_hei_artifacts(payload: dict[str, Any], kind: str, env_id: str, tag: str) -> None:
    """Copy verified checkpoint_final.pt and receipt.json from hei-research artifacts into mirror dir."""
    art_str = payload.get("artifacts")
    if not art_str:
        run_id = payload.get("id") or payload.get("run_id")
        if run_id:
            art_dir = Path.home() / ".local" / "state" / "hei-research" / "runs" / str(run_id) / "artifacts"
        else:
            return
    else:
        art_dir = Path(str(art_str))
    ckpt_src = art_dir / "checkpoint_final.pt"
    receipt_src = art_dir / "receipt.json"
    if not ckpt_src.is_file() or not receipt_src.is_file():
        return
    mirror_dir = REPO_ROOT / "artifacts" / "stage1a-gpu" / ("anchors" if kind == "anchor" else "arms") / env_id
    mirror_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dst = mirror_dir / f"{tag}.pt"
    receipt_dst = mirror_dir / f"{tag}_receipt.json"
    if not ckpt_dst.exists():
        tmp = mirror_dir / f"{tag}.pt.tmp"
        shutil.copy2(ckpt_src, tmp)
        tmp.replace(ckpt_dst)
    if not receipt_dst.exists():
        rec = json.loads(receipt_src.read_text(encoding="utf-8"))
        rec["mirror_checkpoint"] = str(ckpt_dst.relative_to(REPO_ROOT))
        receipt_dst.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")


def _run_hei_recipe(recipe: Path, backend: str, allow_remote: bool, wait_timeout: int) -> dict[str, Any]:
    cmd = [str(HEI_BIN), "--json", "run", "--backend", backend]
    if allow_remote:
        cmd.append("--allow-remote")
    cmd.append(str(recipe))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)  # nosec B603
    if proc.returncode != 0:
        raise RuntimeError(f"hei-research run failed for {recipe.name}: {proc.stderr.strip() or proc.stdout.strip()}")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    run_id = payload.get("id") or payload.get("run_id")
    status = str(payload.get("state") or payload.get("status") or "").lower()
    if run_id and status not in {"completed", "succeeded", "done", "ok"}:
        wait_cmd = [str(HEI_BIN), "--json", "wait", "--timeout", str(wait_timeout), str(run_id)]
        wproc = subprocess.run(wait_cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)  # nosec B603
        if wproc.returncode != 0:
            raise RuntimeError(f"hei-research wait failed for {run_id}: {wproc.stderr.strip() or wproc.stdout.strip()}")
        payload = json.loads(wproc.stdout.strip().splitlines()[-1])
        final_state = str(payload.get("state") or payload.get("status") or "").lower()
        if final_state not in {"completed", "succeeded", "done", "ok"}:
            raise RuntimeError(f"hei-research run {run_id} ended in non-completed state: {payload}")
    return payload


def build_queue(protocol: dict[str, Any], phase: str, env_filter: str) -> list[dict[str, Any]]:
    benchmarks = {b["env_id"]: b for b in protocol["benchmarks"]}
    env_order = ["mo-lunar-lander-v3", "minecart-v0"] if env_filter == "all" else [env_filter]
    items: list[dict[str, Any]] = []

    for env_id in env_order:
        bench = benchmarks[env_id]
        short_env = "minecart" if env_id == "minecart-v0" else "lunar"
        expected_steps = int(bench["training_interactions"])
        if phase == "anchors":
            for v in range(int(bench["anchor_spec"]["vertices"])):
                for seed in bench["anchor_spec"]["anchor_seeds"]:
                    name = f"stage1a-{short_env}-anchor-v{v}-s{seed}"
                    items.append(
                        {
                            "kind": "anchor",
                            "env_id": env_id,
                            "tag": f"{env_id}_anchor_vertex{v}_seed{seed}",
                            "recipe": RECIPE_ROOT / "anchors" / f"{name}.toml",
                            "expected_steps": expected_steps,
                        }
                    )
        else:
            # Strict paired-seed ordering: keep all arms for a given seed adjacent on the same backend.
            for seed in bench["paired_seeds"]:
                for arm in bench["arms_included"]:
                    name = f"stage1a-{short_env}-{arm.replace('_', '-')}-s{seed}"
                    items.append(
                        {
                            "kind": "arm",
                            "env_id": env_id,
                            "tag": f"{env_id}_arm_{arm}_seed{seed}",
                            "recipe": RECIPE_ROOT / "arms" / short_env / f"{name}.toml",
                            "expected_steps": expected_steps,
                        }
                    )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["anchors", "arms"], required=True)
    parser.add_argument("--env", choices=["all", "mo-lunar-lander-v3", "minecart-v0"], default="all")
    parser.add_argument("--backend", choices=["local", "kaggle", "colab", "auto"], default="local")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="Optional cap on runs executed in this invocation.")
    parser.add_argument("--wait-timeout", type=int, default=10800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not protocol.get("execution_authorized"):
        raise SystemExit("Protocol execution_authorized is False.")

    if args.phase == "arms":
        _verify_frozen_anchor_manifest(ANCHOR_MANIFEST)

    queue = build_queue(protocol, args.phase, args.env)
    pending = [
        item
        for item in queue
        if not _is_run_complete(item["kind"], item["env_id"], item["tag"], item["expected_steps"])
    ]
    print(
        f"Stage-1A GPU queue (phase={args.phase}, env={args.env}, backend={args.backend}): "
        f"{len(queue) - len(pending)}/{len(queue)} complete, {len(pending)} pending.",
        flush=True,
    )
    if args.dry_run:
        for idx, item in enumerate(pending, start=1):
            print(f"  [{idx}/{len(pending)}] {item['tag']} -> {item['recipe'].relative_to(REPO_ROOT)}")
        return

    if not HEI_BIN.exists():
        raise SystemExit(f"hei-research binary not found at {HEI_BIN}")

    # Mandatory pre-launch doctor & quota check.
    for sub in ("doctor", "quota"):
        res = subprocess.run([str(HEI_BIN), "--json", sub], cwd=REPO_ROOT, capture_output=True, text=True, check=True)  # nosec B603
        print(f"[hei-research {sub}] {res.stdout.strip()}", flush=True)

    executed = 0
    for idx, item in enumerate(pending, start=1):
        if args.max_runs > 0 and executed >= args.max_runs:
            print(f"Reached --max-runs={args.max_runs}; stopping cleanly.", flush=True)
            break
        t0 = time.monotonic()
        print(f"[{idx}/{len(pending)}] Launching {item['tag']} via hei-research ({args.backend})...", flush=True)
        payload = _run_hei_recipe(item["recipe"], args.backend, args.allow_remote, args.wait_timeout)
        _sync_hei_artifacts(payload, item["kind"], item["env_id"], item["tag"])
        wall = time.monotonic() - t0
        if not _is_run_complete(item["kind"], item["env_id"], item["tag"], item["expected_steps"]):
            raise RuntimeError(
                f"Run {item['tag']} returned from hei-research ({payload}) but verified checkpoint/receipt is missing!"
            )
        executed += 1
        print(f"[{idx}/{len(pending)}] Verified {item['tag']} in {wall / 60.0:.1f} min.", flush=True)

    if args.phase == "anchors":
        freeze_cmd = [sys.executable, str(REPO_ROOT / "scripts" / "freeze_stage1a_anchors.py"), "--allow-partial"]
        subprocess.run(freeze_cmd, cwd=REPO_ROOT, check=True)  # nosec B603
        all_anchors = build_queue(protocol, "anchors", "all")
        if all(_is_run_complete(i["kind"], i["env_id"], i["tag"], i["expected_steps"]) for i in all_anchors):
            subprocess.run(  # nosec B603
                [sys.executable, str(REPO_ROOT / "scripts" / "freeze_stage1a_anchors.py")],
                cwd=REPO_ROOT,
                check=True,
            )


if __name__ == "__main__":
    main()

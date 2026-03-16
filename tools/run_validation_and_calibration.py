#!/usr/bin/env python3
"""Run validation and calibration scripts in one reproducible pipeline.

This wrapper executes the current headless diagnostics and calibration scripts
in a fixed sequence, then writes a compact JSON summary with pass or fail status.

Default sequence:
  1) test_extended_diagnostics.py
  2) test_physics_realistic.py
  3) tools/run_sensitivity_sweep.py
  4) tools/analyze_sensitivity_recommendation.py
  5) tools/analyze_sensitivity_recommendation.py --objective agility
  6) tools/analyze_sensitivity_recommendation.py --objective agility --min-improvement 0.0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "tools" / "validation_pipeline_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diagnostics and calibration pipeline")
    parser.add_argument(
        "--skip-extended",
        action="store_true",
        help="Skip test_extended_diagnostics.py",
    )
    parser.add_argument(
        "--skip-realism",
        action="store_true",
        help="Skip test_physics_realistic.py",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip sensitivity sweep and recommendation generation",
    )
    parser.add_argument(
        "--output",
        default=str(OUT_JSON),
        help="Pipeline summary JSON path",
    )
    return parser.parse_args()


def run_step(name: str, cmd: list[str]) -> dict:
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.time() - start

    status = "pass" if proc.returncode == 0 else "fail"
    print(f"[{status.upper()}] {name} ({elapsed:.1f}s)")
    if proc.returncode != 0:
        # Keep failure output concise in console; full stdout/stderr stored in JSON.
        stderr_tail = "\n".join(proc.stderr.splitlines()[-20:])
        stdout_tail = "\n".join(proc.stdout.splitlines()[-20:])
        if stderr_tail:
            print("  stderr (tail):")
            print(stderr_tail)
        elif stdout_tail:
            print("  stdout (tail):")
            print(stdout_tail)

    return {
        "name": name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "status": status,
        "elapsed_seconds": elapsed,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def pipeline_steps(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    py = sys.executable
    steps: list[tuple[str, list[str]]] = []

    if not args.skip_extended:
        steps.append(("extended_diagnostics", [py, "test_extended_diagnostics.py"]))

    if not args.skip_realism:
        steps.append(("physics_realism", [py, "test_physics_realistic.py"]))

    if not args.skip_sweep:
        steps.extend(
            [
                ("sensitivity_sweep", [py, "tools/run_sensitivity_sweep.py"]),
                (
                    "recommendation_balanced",
                    [py, "tools/analyze_sensitivity_recommendation.py"],
                ),
                (
                    "recommendation_agility",
                    [
                        py,
                        "tools/analyze_sensitivity_recommendation.py",
                        "--objective",
                        "agility",
                        "--out-json",
                        "tools/sensitivity_recommendation_agility.json",
                        "--out-md",
                        "tools/sensitivity_recommendation_agility.md",
                    ],
                ),
                (
                    "recommendation_agility_candidate",
                    [
                        py,
                        "tools/analyze_sensitivity_recommendation.py",
                        "--objective",
                        "agility",
                        "--min-improvement",
                        "0.0",
                        "--out-json",
                        "tools/sensitivity_recommendation_agility_candidate.json",
                        "--out-md",
                        "tools/sensitivity_recommendation_agility_candidate.md",
                    ],
                ),
            ]
        )

    return steps


def main() -> int:
    args = parse_args()
    steps = pipeline_steps(args)
    if not steps:
        print("No steps selected. Nothing to run.")
        return 0

    print("=" * 72)
    print("VALIDATION AND CALIBRATION PIPELINE")
    print("=" * 72)

    start = time.time()
    results = [run_step(name, cmd) for name, cmd in steps]
    elapsed = time.time() - start

    failed = [r for r in results if r["status"] != "pass"]
    summary = {
        "workspace_root": str(ROOT),
        "python_executable": sys.executable,
        "elapsed_seconds": elapsed,
        "overall_status": "pass" if not failed else "fail",
        "failed_count": len(failed),
        "step_count": len(results),
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("-" * 72)
    print(f"Pipeline status: {summary['overall_status'].upper()}")
    print(f"Steps: {summary['step_count']} | Failed: {summary['failed_count']}")
    print(f"Summary: {out_path}")

    return 0 if summary["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

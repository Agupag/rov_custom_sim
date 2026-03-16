#!/usr/bin/env python3
"""Master verification harness that runs all debug modules and aggregates artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from debug.common import (
    ARTIFACTS_ROOT,
    capture_environment_snapshot,
    console,
    ensure_dir,
    make_run_id,
    write_json,
    write_text,
)

MODULES = [
    "debug.debug_startup_and_config",
    "debug.debug_thruster_geometry",
    "debug.debug_control_path",
    "debug.debug_physics_sanity",
    "debug.debug_physics_environment_stress",
    "debug.debug_camera_recording_pipeline",
    "debug.debug_ui_truthfulness",
    "debug.debug_ui_settings_contract",
    "debug.debug_runtime_events_integrity",
    "debug.debug_recording_event_file_correlation",
    "debug.debug_runtime_consistency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full ROV verification harness")
    parser.add_argument("--run-id", default=None, help="Optional run identifier")
    parser.add_argument(
        "--artifacts-root",
        default=str(ARTIFACTS_ROOT),
        help="Root output directory for artifacts",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or make_run_id("full")
    base = ensure_dir(Path(args.artifacts_root) / run_id)

    summary = {
        "run_id": run_id,
        "artifacts_dir": str(base),
        "environment": capture_environment_snapshot(),
        "modules": [],
    }

    write_json(base / "environment.json", summary["environment"])

    exit_code = 0
    for module in MODULES:
        cmd = [
            sys.executable,
            "-m",
            module,
            "--run-id",
            run_id,
            "--artifacts-root",
            str(args.artifacts_root),
        ]
        if args.quiet:
            cmd.append("--quiet")

        console(f"[full] running {module}", quiet=args.quiet)
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent), capture_output=True, text=True, check=False)

        summary["modules"].append(
            {
                "module": module,
                "returncode": proc.returncode,
                "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
                "stderr_tail": "\n".join(proc.stderr.splitlines()[-40:]),
            }
        )
        if proc.returncode != 0:
            exit_code = 1

    fail_count = sum(1 for m in summary["modules"] if m["returncode"] != 0)
    summary["overall_status"] = "pass" if fail_count == 0 else "fail"
    summary["failed_modules"] = fail_count

    write_json(base / "summary.json", summary)

    lines = [
        "# Full Verification Summary",
        "",
        f"- Run ID: {run_id}",
        f"- Overall: {summary['overall_status'].upper()}",
        f"- Failed modules: {fail_count}",
        "",
        "## Modules",
    ]
    for mod in summary["modules"]:
        status = "PASS" if mod["returncode"] == 0 else "FAIL"
        lines.append(f"- {mod['module']}: {status}")
    lines.append("")
    lines.append("See each module folder for detailed JSON and Markdown evidence.")
    write_text(base / "summary.md", "\n".join(lines))

    console(f"[full] summary: {base / 'summary.json'}", quiet=args.quiet)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

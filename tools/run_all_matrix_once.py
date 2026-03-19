#!/usr/bin/env python3
"""One-shot matrix runner for all debug modules, test scripts, and calibration."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"all_matrix_{stamp}"

    out_dir = root / "debug_artifacts" / run_id
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    debug_modules = [
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
        "debug.debug_full_system",
    ]

    steps: list[tuple[str, list[str]]] = []
    for mod in debug_modules:
        steps.append(
            (
                mod,
                [
                    "-m",
                    mod,
                    "--run-id",
                    run_id,
                    "--artifacts-root",
                    str(root / "debug_artifacts"),
                ],
            )
        )

    for test_file in sorted(p.name for p in root.glob("test_*.py")):
        steps.append((test_file, [str(root / test_file)]))

    steps.append(
        (
            "tools/run_validation_and_calibration.py",
            [str(root / "tools" / "run_validation_and_calibration.py")],
        )
    )

    py = "/opt/homebrew/anaconda3/envs/rov_conda/bin/python"
    summary: dict[str, object] = {
        "run_id": run_id,
        "workspace": str(root),
        "python": py,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }

    for idx, (name, args) in enumerate(steps, 1):
        cmd = [py] + args
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, check=False)
        elapsed = time.time() - t0

        safe = name.replace("/", "_").replace(".", "_")
        stdout_log = logs_dir / f"{idx:02d}_{safe}.stdout.log"
        stderr_log = logs_dir / f"{idx:02d}_{safe}.stderr.log"
        stdout_log.write_text(proc.stdout or "", encoding="utf-8")
        stderr_log.write_text(proc.stderr or "", encoding="utf-8")

        print(f"[{idx:02d}/{len(steps):02d}] rc={proc.returncode} {name} ({elapsed:.1f}s)")

        cast_steps = summary["steps"]
        assert isinstance(cast_steps, list)
        cast_steps.append(
            {
                "index": idx,
                "name": name,
                "cmd": cmd,
                "returncode": proc.returncode,
                "elapsed_seconds": round(elapsed, 3),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
            }
        )

    all_steps = summary["steps"]
    assert isinstance(all_steps, list)
    failed = [s["name"] for s in all_steps if s["returncode"] != 0]
    elapsed_total = sum(float(s["elapsed_seconds"]) for s in all_steps)

    summary["elapsed_seconds"] = round(elapsed_total, 3)
    summary["failed"] = failed
    summary["failed_count"] = len(failed)
    summary["overall_status"] = "pass" if not failed else "fail"
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()

    summary_path = out_dir / "matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"RUN_ID={run_id}")
    print(f"SUMMARY={summary_path}")
    print(f"OVERALL={summary['overall_status']} FAILED={summary['failed_count']}")
    for name in failed:
        print(f"FAILED_STEP={name}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

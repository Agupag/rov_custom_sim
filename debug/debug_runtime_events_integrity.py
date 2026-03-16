#!/usr/bin/env python3
"""Runtime event log integrity checks for env-gated instrumentation."""

from __future__ import annotations

import json
import os
from pathlib import Path

from debug.common import CheckResult, console, emit_module_results, init_context, parse_common_args, write_json


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_common_args("Verify runtime JSONL event logging integrity")
    ctx = init_context("runtime_events", args)

    import joystick_panel
    import rov_sim

    event_log = ctx.module_dir / "runtime_events_probe.jsonl"
    os.environ["ROV_DEBUG_EVENTS"] = "1"
    os.environ["ROV_DEBUG_EVENTS_FILE"] = str(event_log)

    # Reinitialize event loggers so they read the probe environment variables.
    rov_sim._init_runtime_events()
    joystick_panel._init_runtime_events()

    rov_sim._evt("startup", "probe_sim_start", mode="integrity")
    rov_sim._evt("control", "probe_cooldown_block", thruster=1, elapsed=0.01)
    joystick_panel._evt("ipc", "probe_shared_ready", slots=32)
    joystick_panel._evt("lifecycle", "probe_panel_started", pid=None)

    rows = _read_jsonl(event_log)
    write_json(ctx.module_dir / "events.json", rows)

    required_sources = {"rov_sim", "joystick_panel"}
    required_categories = {"startup", "control", "ipc", "lifecycle"}
    seen_sources = {str(r.get("source", "")) for r in rows}
    seen_categories = {str(r.get("category", "")) for r in rows}

    results: list[CheckResult] = []
    results.append(
        CheckResult(
            name="event_log_file_written",
            status="pass" if event_log.exists() and len(rows) >= 4 else "fail",
            summary="Checked that env-gated runtime event instrumentation writes JSONL lines.",
            evidence={
                "event_log": str(event_log),
                "exists": event_log.exists(),
                "line_count": len(rows),
            },
        )
    )

    results.append(
        CheckResult(
            name="event_schema_fields_present",
            status="pass" if all({"ts_utc", "source", "category", "event", "fields"}.issubset(set(r.keys())) for r in rows) else "fail",
            summary="Checked required JSONL event schema fields are present on every line.",
            evidence={"required_fields": ["ts_utc", "source", "category", "event", "fields"]},
        )
    )

    missing_sources = sorted(required_sources - seen_sources)
    results.append(
        CheckResult(
            name="both_event_sources_emit",
            status="pass" if not missing_sources else "fail",
            summary="Checked that both simulator and joystick panel instrumentation emit events.",
            evidence={"seen_sources": sorted(seen_sources)},
            errors=[f"missing source: {s}" for s in missing_sources],
        )
    )

    missing_categories = sorted(required_categories - seen_categories)
    results.append(
        CheckResult(
            name="expected_categories_present",
            status="pass" if not missing_categories else "warn",
            summary="Checked representative event categories appear in probe output.",
            evidence={"seen_categories": sorted(seen_categories)},
            warnings=[f"missing category: {c}" for c in missing_categories],
        )
    )

    payload = emit_module_results(ctx, "runtime_events", results)
    console(f"[runtime_events] wrote: {ctx.module_dir / 'results.json'}", quiet=args.quiet)
    console(f"[runtime_events] checks: {payload['counts']}", quiet=args.quiet)
    return 1 if payload["counts"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
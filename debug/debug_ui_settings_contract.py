#!/usr/bin/env python3
"""Verify UI settings menu contract against simulator runtime read path."""

from __future__ import annotations

import re
from pathlib import Path

from debug.common import CheckResult, console, emit_module_results, init_context, parse_common_args


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    args = parse_common_args("Verify UI settings contract and simulator consumption path")
    ctx = init_context("ui_settings", args)

    root = Path(__file__).resolve().parent.parent
    panel_path = root / "joystick_panel.py"
    sim_path = root / "rov_sim.py"

    panel_text = _read(panel_path)
    sim_text = _read(sim_path)

    setting_slots = [
        "SET_THRUST_LEVEL",
        "SET_PROPORTIONAL_MODE",
        "SET_DEPTH_HOLD",
        "SET_HEADING_HOLD",
        "SET_CAM_FOLLOW",
        "SET_CAM_CHASE",
        "SET_TOPDOWN",
        "SET_SHOW_FORCE_VECTORS",
        "SET_THRUSTER_FAILURE",
        "SET_EMERGENCY_SURFACE",
        "CMD_RESET_ROV",
        "SET_TRAIL_ENABLED",
    ]

    panel_published = []
    sim_consumed = []
    missing_panel = []
    missing_sim = []

    for slot in setting_slots:
        panel_has = (f"shared_arr[{slot}]" in panel_text) or (f"_shared[{slot}]" in panel_text)
        sim_has = f"joystick_panel._shared[{slot}]" in sim_text
        if panel_has:
            panel_published.append(slot)
        else:
            missing_panel.append(slot)
        if sim_has:
            sim_consumed.append(slot)
        else:
            missing_sim.append(slot)

    results: list[CheckResult] = []
    results.append(
        CheckResult(
            name="settings_slot_wiring_panel_to_sim",
            status="pass" if not missing_panel and not missing_sim else "fail",
            summary="Checked each settings slot is published by panel code and consumed by simulator runtime path.",
            evidence={
                "expected_slots": setting_slots,
                "panel_published": panel_published,
                "sim_consumed": sim_consumed,
            },
            errors=[*(f"panel missing slot write: {s}" for s in missing_panel), *(f"sim missing slot read: {s}" for s in missing_sim)],
        )
    )

    # Cross-check thrust range contract between UI slider and simulator clamp.
    slider_ok = bool(re.search(r"ttk\.Scale\(.*from_=0\.1,\s*to=1\.0", panel_text, flags=re.DOTALL))
    clamp_ok = "THRUST_LEVEL = clamp(float(_set_thrust), 0.1, 1.0)" in sim_text
    results.append(
        CheckResult(
            name="thrust_range_contract",
            status="pass" if slider_ok and clamp_ok else "fail",
            summary="Checked settings slider range aligns with simulator clamp range.",
            evidence={
                "panel_slider_0p1_to_1p0": slider_ok,
                "sim_clamp_0p1_to_1p0": clamp_ok,
            },
        )
    )

    # Reset command semantics should be monotonic pulse-like (only reacts on increment).
    reset_logic_ok = "if _set_reset_cmd > _panel_last_reset_cmd:" in sim_text
    reset_publisher_ok = "_settings_state[\"reset_seq\"] += 1" in panel_text and "shared_arr[CMD_RESET_ROV]" in panel_text
    results.append(
        CheckResult(
            name="reset_command_pulse_semantics",
            status="pass" if reset_logic_ok and reset_publisher_ok else "fail",
            summary="Checked reset command uses increment-only pulse semantics from panel to simulator.",
            evidence={
                "panel_increments_reset_seq": reset_publisher_ok,
                "sim_uses_gt_last_seen": reset_logic_ok,
            },
        )
    )

    # Assist mode feedback indicators are expected to be written back to panel shared memory.
    assist_feedback_ok = (
        "joystick_panel._shared[DEPTH_HOLD_ACTIVE]" in sim_text
        and "joystick_panel._shared[HEADING_HOLD_ACTIVE]" in sim_text
        and "joystick_panel._shared[CONTROL_MODE]" in sim_text
    )
    results.append(
        CheckResult(
            name="settings_feedback_to_ui",
            status="pass" if assist_feedback_ok else "warn",
            summary="Checked simulator writes assist/control status back to shared memory for panel truthfulness.",
            evidence={"feedback_paths_present": assist_feedback_ok},
            warnings=[] if assist_feedback_ok else ["One or more feedback slots missing in simulator write-back path"],
        )
    )

    payload = emit_module_results(ctx, "ui_settings", results)
    console(f"[ui_settings] wrote: {ctx.module_dir / 'results.json'}", quiet=args.quiet)
    console(f"[ui_settings] checks: {payload['counts']}", quiet=args.quiet)
    return 1 if payload["counts"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

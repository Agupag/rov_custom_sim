#!/usr/bin/env python3
"""UI/shared-memory truthfulness checks for panel-visible state versus backend contract."""

from __future__ import annotations

import ctypes

from debug.common import CheckResult, console, emit_module_results, init_context, parse_common_args, write_json


def main() -> int:
    args = parse_common_args("Verify shared-memory/UI contract truthfulness checks")
    ctx = init_context("ui", args)

    import joystick_panel
    import sim_shared

    results: list[CheckResult] = []

    # 1) slot count and reserved-slot contract
    slot_ok = sim_shared.SHARED_SLOT_COUNT == 32 and sim_shared.RESERVED_SLOTS == (29, 30, 31)
    results.append(
        CheckResult(
            name="shared_memory_contract_shape",
            status="pass" if slot_ok else "fail",
            summary="Checked shared-memory slot count and reserved slot contract.",
            evidence={
                "slot_count": sim_shared.SHARED_SLOT_COUNT,
                "reserved_slots": list(sim_shared.RESERVED_SLOTS),
            },
        )
    )

    # 2) label mapping truthfulness for control and recording states
    mode_labels = {
        "binary": sim_shared.control_mode_label(sim_shared.CONTROL_MODE_BINARY),
        "proportional": sim_shared.control_mode_label(sim_shared.CONTROL_MODE_PROPORTIONAL),
    }
    rec_labels = {
        "ok": sim_shared.recording_status_label(sim_shared.REC_STATUS_OK),
        "missing_deps": sim_shared.recording_status_label(sim_shared.REC_STATUS_MISSING_DEPS),
        "writer_open_failed": sim_shared.recording_status_label(sim_shared.REC_STATUS_WRITER_OPEN_FAILED),
        "panel_capture_unavailable": sim_shared.recording_status_label(sim_shared.REC_STATUS_PANEL_CAPTURE_UNAVAILABLE),
        "frame_write_failed": sim_shared.recording_status_label(sim_shared.REC_STATUS_FRAME_WRITE_FAILED),
    }
    labels_ok = mode_labels["binary"] == "BIN" and mode_labels["proportional"] == "PROP"
    results.append(
        CheckResult(
            name="ui_label_mapping_consistency",
            status="pass" if labels_ok else "warn",
            summary="Checked status label helpers that drive panel-visible mode and recording text.",
            evidence={"mode_labels": mode_labels, "recording_labels": rec_labels},
        )
    )

    # 3) panel write/read truthfulness for joystick state path
    joystick_panel._ensure_shared()
    with joystick_panel._shared.get_lock():
        joystick_panel._shared[sim_shared.SURGE] = 0.45
        joystick_panel._shared[sim_shared.YAW] = -0.55
        joystick_panel._shared[sim_shared.HEAVE] = 1.0
        joystick_panel._shared[sim_shared.ACTIVE] = 1.0
        joystick_panel._shared[sim_shared.CAM_TILT] = -0.25

    state = joystick_panel.get_joystick_state()
    path_ok = (
        abs(state.get("surge", 0.0) - 0.45) < 1e-6
        and abs(state.get("yaw", 0.0) + 0.55) < 1e-6
        and abs(state.get("heave", 0.0) - 1.0) < 1e-6
        and bool(state.get("active", False))
        and abs(state.get("cam_tilt", 0.0) + 0.25) < 1e-6
    )
    results.append(
        CheckResult(
            name="shared_memory_roundtrip_state",
            status="pass" if path_ok else "fail",
            summary="Verified panel shared-memory write/read roundtrip for joystick-visible fields.",
            evidence={"state": state},
        )
    )

    # 4) recording status roundtrip for panel banner/toast signal source
    with joystick_panel._shared.get_lock():
        joystick_panel._shared[sim_shared.REC_STATUS] = sim_shared.REC_STATUS_FRAME_WRITE_FAILED
    rec_status = joystick_panel.get_recording_status()
    rec_ok = abs(rec_status - sim_shared.REC_STATUS_FRAME_WRITE_FAILED) < 1e-6
    results.append(
        CheckResult(
            name="recording_status_signal_path",
            status="pass" if rec_ok else "fail",
            summary="Verified REC_STATUS shared slot propagates through panel helper path.",
            evidence={"read_status": rec_status, "expected_status": sim_shared.REC_STATUS_FRAME_WRITE_FAILED},
        )
    )

    # 5) panel screenshot buffer signal path truthfulness
    with joystick_panel._panel_seq.get_lock():
        seq_before = int(joystick_panel._panel_seq.value)

    pattern = bytes([17, 33, 99]) * (sim_shared.CTRL_W * sim_shared.CTRL_H)
    ctypes.memmove(ctypes.addressof(joystick_panel._panel_buf), pattern, len(pattern))
    with joystick_panel._panel_seq.get_lock():
        joystick_panel._panel_seq.value += 1

    seq_after, panel_bytes = joystick_panel.get_panel_frame()
    panel_path_ok = (
        seq_after >= seq_before + 1
        and panel_bytes is not None
        and len(panel_bytes) == (sim_shared.CTRL_W * sim_shared.CTRL_H * 3)
        and panel_bytes[:24] == pattern[:24]
    )
    results.append(
        CheckResult(
            name="panel_frame_signal_path",
            status="pass" if panel_path_ok else "fail",
            summary="Verified panel RGB screenshot bytes propagate through shared buffer and read helper path.",
            evidence={
                "seq_before": seq_before,
                "seq_after": seq_after,
                "panel_bytes_len": len(panel_bytes) if panel_bytes is not None else 0,
                "prefix_matches": bool(panel_bytes is not None and panel_bytes[:24] == pattern[:24]),
            },
        )
    )

    payload = emit_module_results(ctx, "ui", results)
    write_json(
        ctx.module_dir / "state_snapshot.json",
        {
            "joystick_state": state,
            "recording_status": rec_status,
            "panel_seq_before": seq_before,
            "panel_seq_after": seq_after,
        },
    )
    console(f"[ui] wrote: {ctx.module_dir / 'results.json'}", quiet=args.quiet)
    console(f"[ui] checks: {payload['counts']}", quiet=args.quiet)
    return 1 if payload["counts"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Correlate recording lifecycle events with tangible recording file artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from debug.common import CheckResult, console, emit_module_results, init_context, parse_common_args, write_json


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> int:
    args = parse_common_args("Verify recording lifecycle events correlate with file artifacts")
    ctx = init_context("recording_correlation", args)

    import rov_sim
    import joystick_panel

    event_log = ctx.module_dir / "recording_events.jsonl"
    rec_probe = ctx.module_dir / "recording_probe_event_corr.mp4"

    os.environ["ROV_DEBUG_EVENTS"] = "1"
    os.environ["ROV_DEBUG_EVENTS_FILE"] = str(event_log)
    rov_sim._init_runtime_events()
    joystick_panel._init_runtime_events()

    has_cv2 = bool(getattr(rov_sim, "cv2", None) is not None)
    has_numpy = bool(getattr(rov_sim, "HAS_NUMPY", False))

    results: list[CheckResult] = []

    if has_cv2 and has_numpy:
        cv2_mod = rov_sim.cv2
        np_mod = rov_sim.np
        out_w = int(rov_sim.REC_3D_W + joystick_panel.CTRL_W)
        out_h = int(max(rov_sim.REC_3D_H, joystick_panel.CTRL_H))
        frame0 = np_mod.zeros((out_h, out_w, 3), dtype=np_mod.uint8)
        frame1 = np_mod.zeros((out_h, out_w, 3), dtype=np_mod.uint8)
        frame0[:, : rov_sim.REC_3D_W, 0] = 160
        frame1[:, rov_sim.REC_3D_W :, 1] = 180

        writer_ok = False
        err = None
        try:
            fourcc = cv2_mod.VideoWriter_fourcc(*"mp4v")
            writer = cv2_mod.VideoWriter(str(rec_probe), fourcc, float(rov_sim.REC_FPS), (out_w, out_h))
            writer_ok = bool(writer.isOpened())
            if writer_ok:
                rov_sim._evt("recording", "start_ok", path=str(rec_probe), out_w=out_w, out_h=out_h, fps=rov_sim.REC_FPS)
                writer.write(frame0)
                writer.write(frame1)
                writer.release()
                rov_sim._evt("recording", "stop_ok", path=str(rec_probe), frames=2, duration_s=round(2.0 / float(max(1, rov_sim.REC_FPS)), 3))
            else:
                writer.release()
                rov_sim._evt("recording", "start_failed_writer_open", path=str(rec_probe))
        except Exception as exc:
            err = str(exc)
            rov_sim._evt("recording", "frame_write_error", error=err, frame_count=0)

        size_bytes = int(rec_probe.stat().st_size) if rec_probe.exists() else 0
        rows = _read_jsonl(event_log)
        write_json(ctx.module_dir / "events.json", rows)

        start_events = [r for r in rows if r.get("category") == "recording" and r.get("event") == "start_ok"]
        stop_events = [r for r in rows if r.get("category") == "recording" and r.get("event") == "stop_ok"]

        path_start_match = any(r.get("fields", {}).get("path") == str(rec_probe) for r in start_events)
        path_stop_match = any(r.get("fields", {}).get("path") == str(rec_probe) for r in stop_events)

        results.append(
            CheckResult(
                name="recording_probe_file_written",
                status="pass" if writer_ok and size_bytes > 1024 else "fail",
                summary="Checked recording probe wrote a non-empty MP4 artifact.",
                evidence={"probe_path": str(rec_probe), "writer_ok": writer_ok, "size_bytes": size_bytes},
                errors=[err] if err else [],
            )
        )
        results.append(
            CheckResult(
                name="recording_events_match_artifact_path",
                status="pass" if path_start_match and path_stop_match else "fail",
                summary="Checked recording start/stop events reference the same file artifact path.",
                evidence={
                    "event_log": str(event_log),
                    "start_event_count": len(start_events),
                    "stop_event_count": len(stop_events),
                    "start_path_match": path_start_match,
                    "stop_path_match": path_stop_match,
                },
            )
        )
    else:
        results.append(
            CheckResult(
                name="recording_probe_dependencies",
                status="warn",
                summary="Skipped recording event/file correlation because recording dependencies are unavailable in this interpreter.",
                evidence={"has_cv2": has_cv2, "has_numpy": has_numpy},
                warnings=["Install/use environment with numpy + opencv to enable recording correlation check"],
            )
        )

    payload = emit_module_results(ctx, "recording_correlation", results)
    console(f"[recording_correlation] wrote: {ctx.module_dir / 'results.json'}", quiet=args.quiet)
    console(f"[recording_correlation] checks: {payload['counts']}", quiet=args.quiet)
    return 1 if payload["counts"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

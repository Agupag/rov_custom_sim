#!/usr/bin/env python3
"""Camera, preview, and recording pipeline verification checks."""

from __future__ import annotations

import os
from pathlib import Path

import pybullet as p
import pybullet_data

from debug.common import CheckResult, console, emit_module_results, init_context, parse_common_args, write_json


def main() -> int:
    args = parse_common_args("Verify camera pose source, frame freshness, and recording prerequisites")
    ctx = init_context("camera_recording", args)

    import rov_sim
    import joystick_panel

    results: list[CheckResult] = []

    # 1) camera pose source check
    center, _size = rov_sim.obj_bounds(rov_sim.OBJ_FILE)
    cam_info = rov_sim.find_camera_pose_from_gltf(rov_sim.GLTF_FILE, center)
    results.append(
        CheckResult(
            name="camera_pose_source",
            status="pass" if cam_info is not None else "warn",
            summary="Checked whether camera pose was discovered from active GLTF.",
            evidence={
                "active_gltf": rov_sim.GLTF_FILE,
                "camera_pose_from_gltf": cam_info is not None,
                "fallback_pose_would_be_used": cam_info is None,
            },
        )
    )

    # 2) frame buffer freshness check via shared memory sequence increments
    joystick_panel._ensure_shared()
    seq0 = joystick_panel._frame_seq.value if joystick_panel._frame_seq is not None else -1
    fake_a = bytes([10]) * (joystick_panel.CAM_W * joystick_panel.CAM_H * 3)
    fake_b = bytes([200]) * (joystick_panel.CAM_W * joystick_panel.CAM_H * 3)
    joystick_panel.push_camera_frame(fake_a)
    joystick_panel.push_camera_frame(fake_b)
    seq1 = joystick_panel._frame_seq.value if joystick_panel._frame_seq is not None else -1

    results.append(
        CheckResult(
            name="preview_frame_seq_progress",
            status="pass" if seq1 >= seq0 + 2 else "fail",
            summary="Verified camera frame writes advance shared frame sequence counter.",
            evidence={"seq_before": seq0, "seq_after": seq1, "delta": seq1 - seq0},
        )
    )

    # 3) recording dependency readiness check
    dep_ok = bool(rov_sim.HAS_NUMPY and (rov_sim.cv2 is not None))
    rec_dir_exists = Path(rov_sim.REC_SAVE_DIR).exists()
    warnings = []
    if not dep_ok:
        warnings.append("opencv-python and/or numpy missing; recording start cannot be fully verified")
    if not rec_dir_exists:
        warnings.append("REC_SAVE_DIR does not exist")

    results.append(
        CheckResult(
            name="recording_prerequisites",
            status="pass" if dep_ok and rec_dir_exists else "warn",
            summary="Checked recording dependency and output directory prerequisites.",
            evidence={
                "has_numpy": rov_sim.HAS_NUMPY,
                "has_cv2": rov_sim.cv2 is not None,
                "rec_save_dir": rov_sim.REC_SAVE_DIR,
                "rec_save_dir_exists": rec_dir_exists,
                "rec_fps": rov_sim.REC_FPS,
            },
            warnings=warnings,
        )
    )

    # 3b) recording file integrity check (encode + decode probe clip)
    if dep_ok and rec_dir_exists:
        cv2_mod = getattr(rov_sim, "cv2", None)
        np_mod = getattr(rov_sim, "np", None)
        rec_probe_path = ctx.module_dir / "recording_probe.mp4"
        rec_err = None
        writer_opened = False
        decode_ok = False
        frame_delta_mean = None
        file_size = 0
        frame_count = 0

        try:
            out_w = int(rov_sim.REC_3D_W + joystick_panel.CTRL_W)
            out_h = int(max(rov_sim.REC_3D_H, joystick_panel.CTRL_H))
            frame_a = np_mod.zeros((out_h, out_w, 3), dtype=np_mod.uint8)
            frame_b = np_mod.zeros((out_h, out_w, 3), dtype=np_mod.uint8)
            frame_a[:, : rov_sim.REC_3D_W, 1] = 180
            frame_b[:, rov_sim.REC_3D_W :, 2] = 200

            fourcc = cv2_mod.VideoWriter_fourcc(*"mp4v")
            writer = cv2_mod.VideoWriter(str(rec_probe_path), fourcc, float(rov_sim.REC_FPS), (out_w, out_h))
            writer_opened = bool(writer.isOpened())
            if writer_opened:
                writer.write(frame_a)
                writer.write(frame_b)
            writer.release()

            if rec_probe_path.exists():
                file_size = int(rec_probe_path.stat().st_size)

            cap = cv2_mod.VideoCapture(str(rec_probe_path))
            ret1, dec_a = cap.read()
            ret2, dec_b = cap.read()
            frame_count = int(cap.get(cv2_mod.CAP_PROP_FRAME_COUNT))
            cap.release()

            decode_ok = bool(ret1 and ret2 and dec_a is not None and dec_b is not None)
            if decode_ok:
                diff = cv2_mod.absdiff(dec_a, dec_b)
                frame_delta_mean = float(diff.mean())
        except Exception as exc:
            rec_err = str(exc)

        rec_ok = writer_opened and file_size > 1024 and decode_ok and (frame_delta_mean is not None and frame_delta_mean > 0.5)
        results.append(
            CheckResult(
                name="recording_file_integrity",
                status="pass" if rec_ok else "fail",
                summary="Verified recording probe can be encoded, decoded, and shows frame-to-frame content changes.",
                evidence={
                    "probe_path": str(rec_probe_path),
                    "writer_opened": writer_opened,
                    "file_size_bytes": file_size,
                    "decoded_two_frames": decode_ok,
                    "decoded_frame_count": frame_count,
                    "mean_abs_frame_delta": frame_delta_mean,
                },
                errors=[rec_err] if rec_err else [],
            )
        )
    else:
        results.append(
            CheckResult(
                name="recording_file_integrity",
                status="warn",
                summary="Skipped recording file integrity probe because prerequisites are not available.",
                evidence={
                    "has_numpy": rov_sim.HAS_NUMPY,
                    "has_cv2": rov_sim.cv2 is not None,
                    "rec_save_dir_exists": rec_dir_exists,
                },
                warnings=["Install numpy/opencv and ensure REC_SAVE_DIR exists to enable this check"],
            )
        )

    # 4) 3D camera capture sanity using active camera renderer in DIRECT mode
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -rov_sim.GRAVITY)
    p.loadURDF("plane.urdf", [0, 0, rov_sim.SEABED_Z])
    rov, _mesh_center = rov_sim.build_rov()
    base_pos, base_quat = p.getBasePositionAndOrientation(rov)

    cam_pos_body = (0.18, 0.0, 0.05) if cam_info is None else cam_info[0]
    cam_fwd_body = (1.0, 0.0, 0.0) if cam_info is None else cam_info[1]
    cam_up_body = (0.0, 0.0, 1.0) if cam_info is None else cam_info[2]

    cam_pos_world_rel = p.rotateVector(base_quat, cam_pos_body)
    cam_pos_world = (
        base_pos[0] + cam_pos_world_rel[0],
        base_pos[1] + cam_pos_world_rel[1],
        base_pos[2] + cam_pos_world_rel[2],
    )
    cam_fwd_world = p.rotateVector(base_quat, cam_fwd_body)
    cam_up_world = p.rotateVector(base_quat, cam_up_body)
    cam_target = (
        cam_pos_world[0] + cam_fwd_world[0],
        cam_pos_world[1] + cam_fwd_world[1],
        cam_pos_world[2] + cam_fwd_world[2],
    )

    view = p.computeViewMatrix(cam_pos_world, cam_target, cam_up_world)
    proj = p.computeProjectionMatrixFOV(
        fov=rov_sim.CAM_FOV,
        aspect=float(rov_sim.CAM_PREVIEW_W) / float(rov_sim.CAM_PREVIEW_H),
        nearVal=0.08,
        farVal=rov_sim.CAM_FAR,
    )
    img = p.getCameraImage(rov_sim.CAM_PREVIEW_W, rov_sim.CAM_PREVIEW_H, viewMatrix=view, projectionMatrix=proj, renderer=rov_sim.CAM_RENDERER)
    p.disconnect(cid)

    capture_ok = img is not None and img[2] is not None
    results.append(
        CheckResult(
            name="direct_camera_capture",
            status="pass" if capture_ok else "fail",
            summary="Checked DIRECT-mode camera capture returns pixel data using active renderer.",
            evidence={"capture_ok": capture_ok, "renderer": int(rov_sim.CAM_RENDERER)},
        )
    )

    payload = emit_module_results(ctx, "camera_recording", results)
    write_json(ctx.module_dir / "snapshot.json", {
        "active_obj": rov_sim.OBJ_FILE,
        "active_gltf": rov_sim.GLTF_FILE,
        "camera_pose_from_gltf": cam_info is not None,
    })
    console(f"[camera_recording] wrote: {ctx.module_dir / 'results.json'}", quiet=args.quiet)
    console(f"[camera_recording] checks: {payload['counts']}", quiet=args.quiet)
    return 1 if payload["counts"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

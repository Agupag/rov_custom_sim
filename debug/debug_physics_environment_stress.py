#!/usr/bin/env python3
"""Stress-check physics and environment presets for instability and invalid states."""

from __future__ import annotations

import math
import random

import pybullet as p
import pybullet_data

from debug.common import CheckResult, console, emit_module_results, init_context, parse_common_args, write_json


def _is_finite_vec(v) -> bool:
    return all(math.isfinite(float(x)) for x in v)


def _step_world(rov_sim, rov, thrusters, thr_cmd, thr_level):
    base_pos, base_quat = p.getBasePositionAndOrientation(rov)
    lin, ang = p.getBaseVelocity(rov)

    depth = max(0.0, rov_sim.SURFACE_Z - base_pos[2])
    hull_half_z = 0.15
    submersion = depth / hull_half_z if depth < hull_half_z else 1.0

    depth_buoyancy_factor = max(0.5, 1.0 - rov_sim.DEPTH_BUOYANCY_COMPRESSIBILITY * depth)
    buoy_force = rov_sim.MASS * rov_sim.GRAVITY * rov_sim.BUOYANCY_SCALE * depth_buoyancy_factor * submersion

    cob_rel_world = p.rotateVector(base_quat, rov_sim.COB_OFFSET_BODY)
    cob_world = (base_pos[0] + cob_rel_world[0], base_pos[1] + cob_rel_world[1], base_pos[2] + cob_rel_world[2])
    p.applyExternalForce(rov, -1, (0.0, 0.0, buoy_force), cob_world, p.WORLD_FRAME)

    rov_sim.apply_ballast(rov, base_pos, base_quat)
    if submersion > 0.01:
        rov_sim.apply_righting_torque(rov, base_quat, ang, submersion=submersion)
    rov_sim.apply_drag(rov, base_pos, base_quat, lin, ang)

    inv_q = p.invertTransform([0, 0, 0], base_quat)[1]
    v_body = p.rotateVector(inv_q, lin)

    for i, thr in enumerate(thrusters):
        if thr_cmd[i] > thr_level[i]:
            tau = rov_sim.THRUSTER_TAU_UP
        elif thr_cmd[i] < thr_level[i]:
            tau = rov_sim.THRUSTER_TAU_DN
        else:
            tau = rov_sim.THRUSTER_TAU_DN

        thr_level[i] += (rov_sim.DT / max(1e-6, tau)) * (thr_cmd[i] - thr_level[i])
        thr_level[i] = max(-1.0, min(1.0, thr_level[i]))
        if abs(thr_level[i]) < 1e-4:
            thr_level[i] = 0.0
        if abs(thr_level[i]) <= 1e-4:
            continue

        thrust_max = rov_sim.MAX_THRUST_H if thr["kind"] == "H" else rov_sim.MAX_THRUST_V
        thrust = thrust_max * thr_level[i] * rov_sim.THRUST_LEVEL
        if thrust < 0.0:
            thrust *= rov_sim.BACKWARDS_THRUST_SCALE

        speed_along = v_body[0] * thr["dir"][0] + v_body[1] * thr["dir"][1] + v_body[2] * thr["dir"][2]
        loss = max(0.0, min(0.9, rov_sim.THRUSTER_SPEED_LOSS_COEF * abs(speed_along)))
        thrust *= (1.0 - loss)

        dir_world = p.rotateVector(base_quat, thr["dir"])
        force = (dir_world[0] * thrust, dir_world[1] * thrust, dir_world[2] * thrust)
        rel_pos_world = p.rotateVector(base_quat, thr["pos"])
        world_pos = (base_pos[0] + rel_pos_world[0], base_pos[1] + rel_pos_world[1], base_pos[2] + rel_pos_world[2])
        p.applyExternalForce(rov, -1, force, world_pos, p.WORLD_FRAME)

    p.stepSimulation()


def _run_one_preset(rov_sim, preset_key: str, steps: int = 900) -> dict:
    rng = random.Random(123 + len(preset_key))
    p.resetSimulation()
    p.setGravity(0, 0, -rov_sim.GRAVITY)
    p.setPhysicsEngineParameter(fixedTimeStep=rov_sim.DT, numSolverIterations=50, numSubSteps=1)

    env_ids = rov_sim.create_environment(preset_key)
    obstacles = rov_sim.spawn_obstacles(rov_sim.NUM_OBSTACLES)

    rov, center = rov_sim.build_rov()
    auto_thr = rov_sim.detect_thrusters_from_gltf(rov_sim.GLTF_FILE, center)
    thrusters = auto_thr if auto_thr else rov_sim.THRUSTERS
    n_thr = len(thrusters)
    thr_cmd = [0.0] * n_thr
    thr_level = [0.0] * n_thr

    max_speed = 0.0
    max_abs_omega = 0.0
    invalid_state_step = None
    obstacle_invalid = 0
    sampled = []

    for step in range(steps):
        if step % 20 == 0:
            for i in range(n_thr):
                thr_cmd[i] = rng.uniform(-1.0, 1.0)

        _step_world(rov_sim, rov, thrusters, thr_cmd, thr_level)
        rov_sim.apply_obstacle_water_forces(obstacles)

        pos, quat = p.getBasePositionAndOrientation(rov)
        lin, ang = p.getBaseVelocity(rov)

        if (not _is_finite_vec(pos)) or (not _is_finite_vec(quat)) or (not _is_finite_vec(lin)) or (not _is_finite_vec(ang)):
            invalid_state_step = step
            break

        speed = math.sqrt(lin[0] * lin[0] + lin[1] * lin[1] + lin[2] * lin[2])
        omega = max(abs(float(ang[0])), abs(float(ang[1])), abs(float(ang[2])))
        max_speed = max(max_speed, speed)
        max_abs_omega = max(max_abs_omega, omega)

        if step % 100 == 0:
            sampled.append({
                "step": step,
                "pos": [round(float(x), 4) for x in pos],
                "speed": round(float(speed), 5),
                "omega_max": round(float(omega), 5),
            })

        for oid in obstacles:
            try:
                op, _oq = p.getBasePositionAndOrientation(oid)
                ov, _ow = p.getBaseVelocity(oid)
                if (not _is_finite_vec(op)) or (not _is_finite_vec(ov)):
                    obstacle_invalid += 1
            except p.error:
                obstacle_invalid += 1

    return {
        "preset": preset_key,
        "env_count": len(env_ids),
        "obstacle_count": len(obstacles),
        "thruster_count": n_thr,
        "invalid_state_step": invalid_state_step,
        "obstacle_invalid_count": obstacle_invalid,
        "max_speed_mps": max_speed,
        "max_abs_omega": max_abs_omega,
        "samples": sampled,
    }


def main() -> int:
    args = parse_common_args("Stress-check physics and environment stability under adversarial commands")
    ctx = init_context("physics_env_stress", args)

    import rov_sim

    # Keep runs deterministic/headless.
    rov_sim.SLEEP_REALTIME = False
    rov_sim.HUD_ENABLED = False
    rov_sim.ENABLE_MARKERS = False
    rov_sim.ENABLE_CAMERA_PREVIEW = False
    rov_sim.ENABLE_THRUSTER_ARROWS = False

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    reports = []
    for preset in sorted(rov_sim.ENVIRONMENT_PRESETS.keys()):
        reports.append(_run_one_preset(rov_sim, preset, steps=900))

    p.disconnect(cid)

    write_json(ctx.module_dir / "trace.json", reports)

    any_invalid = any(r["invalid_state_step"] is not None for r in reports)
    any_obs_invalid = any(r["obstacle_invalid_count"] > 0 for r in reports)
    max_speed = max((r["max_speed_mps"] for r in reports), default=0.0)
    max_omega = max((r["max_abs_omega"] for r in reports), default=0.0)

    results: list[CheckResult] = []
    results.append(
        CheckResult(
            name="rov_state_finite_under_stress",
            status="pass" if not any_invalid else "fail",
            summary="Checked ROV state remains finite (no NaN/Inf) across environment presets under adversarial command changes.",
            evidence={
                "preset_invalid_steps": {r["preset"]: r["invalid_state_step"] for r in reports},
            },
        )
    )
    results.append(
        CheckResult(
            name="obstacle_state_finite_under_stress",
            status="pass" if not any_obs_invalid else "warn",
            summary="Checked obstacle state updates remain finite during water-force integration.",
            evidence={
                "obstacle_invalid_counts": {r["preset"]: r["obstacle_invalid_count"] for r in reports},
            },
            warnings=[] if not any_obs_invalid else ["One or more obstacle state reads were invalid during stress run"],
        )
    )
    results.append(
        CheckResult(
            name="bounded_kinematics_under_stress",
            status="pass" if (max_speed < 6.0 and max_omega < 15.0) else "warn",
            summary="Checked speed and angular-rate peaks stay within conservative sanity bounds.",
            evidence={"max_speed_mps": max_speed, "max_abs_omega": max_omega},
            warnings=[] if (max_speed < 6.0 and max_omega < 15.0) else ["Kinematics exceeded conservative stress bound"],
        )
    )

    payload = emit_module_results(ctx, "physics_env_stress", results)
    console(f"[physics_env_stress] wrote: {ctx.module_dir / 'results.json'}", quiet=args.quiet)
    console(f"[physics_env_stress] checks: {payload['counts']}", quiet=args.quiet)
    return 1 if payload["counts"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

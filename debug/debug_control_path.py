#!/usr/bin/env python3
"""End-to-end control path verification from shared memory to applied thrust."""

from __future__ import annotations

import math
import time

import pybullet as p
import pybullet_data

from debug.common import CheckResult, console, emit_module_results, init_context, parse_common_args, write_json
from debug.scenarios import DEFAULT_CONTROL_PATH_SCENARIO


def _simulate_control_loop(rov_sim, joystick_panel):
    # Shared memory init
    joystick_panel._ensure_shared()
    with joystick_panel._shared.get_lock():
        for idx in range(joystick_panel.SHARED_SLOT_COUNT):
            joystick_panel._shared[idx] = 0.0
        joystick_panel._shared[joystick_panel.ACTIVE] = 1.0

    # World setup
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -rov_sim.GRAVITY)
    p.setPhysicsEngineParameter(fixedTimeStep=rov_sim.DT, numSolverIterations=50, numSubSteps=1)
    p.loadURDF("plane.urdf", [0, 0, rov_sim.SEABED_Z])

    rov, mesh_center = rov_sim.build_rov()
    auto_thr = rov_sim.detect_thrusters_from_gltf(rov_sim.GLTF_FILE, mesh_center)
    thrusters = auto_thr if auto_thr else rov_sim.THRUSTERS

    thr_cmd = [0.0] * len(thrusters)
    thr_level = [0.0] * len(thrusters)
    js_last_sign = [0] * len(thrusters)
    js_last_switch_t = [0.0] * len(thrusters)

    traces = []
    cooldown_block_count = 0
    accepted_flip_count = 0

    steps_total = sum(int(round(ph.duration_s / rov_sim.DT)) for ph in DEFAULT_CONTROL_PATH_SCENARIO)
    scenario_steps = []
    for ph in DEFAULT_CONTROL_PATH_SCENARIO:
        scenario_steps.extend([ph] * int(round(ph.duration_s / rov_sim.DT)))

    if len(scenario_steps) < steps_total:
        scenario_steps.extend([DEFAULT_CONTROL_PATH_SCENARIO[-1]] * (steps_total - len(scenario_steps)))

    for step in range(steps_total):
        ph = scenario_steps[step]

        with joystick_panel._shared.get_lock():
            joystick_panel._shared[joystick_panel.SURGE] = ph.surge
            joystick_panel._shared[joystick_panel.YAW] = ph.yaw
            joystick_panel._shared[joystick_panel.HEAVE] = ph.heave
            joystick_panel._shared[joystick_panel.ACTIVE] = ph.active

        js = joystick_panel.get_joystick_state()
        js_cmds = joystick_panel.mix_joystick_to_thruster_cmds(
            js,
            len(thrusters),
            proportional=rov_sim.PROPORTIONAL_MODE,
            input_exponent=rov_sim.INPUT_CURVE_EXPONENT,
            input_deadzone=rov_sim.INPUT_DEADZONE,
        )

        now = time.monotonic()
        for i in range(len(thrusters)):
            desired = js_cmds[i]
            if abs(desired) < 0.03:
                desired_sign = 0
            elif desired > 0:
                desired_sign = 1
            else:
                desired_sign = -1

            if desired_sign != 0 and js_last_sign[i] != 0 and desired_sign != js_last_sign[i]:
                if (now - js_last_switch_t[i]) < rov_sim.JOYSTICK_SWITCH_COOLDOWN:
                    desired = 0.0
                    desired_sign = 0
                    cooldown_block_count += 1
                else:
                    js_last_switch_t[i] = now
                    accepted_flip_count += 1
            elif desired_sign != js_last_sign[i]:
                js_last_switch_t[i] = now

            js_last_sign[i] = desired_sign
            thr_cmd[i] = 0.0 if abs(desired) < 0.03 else desired

        if len(thrusters) > 2:
            heave = js.get("heave", 0.0)
            if abs(heave) > 0.5:
                thr_cmd[2] = 1.0 if heave > 0 else -1.0
            else:
                thr_cmd[2] = 0.0

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

        net_force = [0.0, 0.0, 0.0]
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

            net_force[0] += force[0]
            net_force[1] += force[1]
            net_force[2] += force[2]

        p.stepSimulation()

        if step % max(1, int(round(0.1 / rov_sim.DT))) == 0:
            lin_new, ang_new = p.getBaseVelocity(rov)
            traces.append(
                {
                    "t": round(step * rov_sim.DT, 4),
                    "phase": ph.name,
                    "shared": {
                        "surge": js.get("surge", 0.0),
                        "yaw": js.get("yaw", 0.0),
                        "heave": js.get("heave", 0.0),
                        "active": js.get("active", False),
                    },
                    "mixer_cmds": [round(x, 4) for x in js_cmds],
                    "thr_cmd": [round(x, 4) for x in thr_cmd],
                    "thr_level": [round(x, 4) for x in thr_level],
                    "net_force_world": [round(x, 5) for x in net_force],
                    "lin_vel_world": [round(x, 5) for x in lin_new],
                    "ang_vel_world": [round(x, 5) for x in ang_new],
                }
            )

    end_pos, _ = p.getBasePositionAndOrientation(rov)
    p.disconnect(cid)

    return {
        "trace": traces,
        "cooldown_block_count": cooldown_block_count,
        "accepted_flip_count": accepted_flip_count,
        "final_position": [round(end_pos[0], 5), round(end_pos[1], 5), round(end_pos[2], 5)],
        "thruster_count": len(thrusters),
    }


def main() -> int:
    args = parse_common_args("Verify control path from shared memory to applied thrust")
    ctx = init_context("control_path", args)

    import rov_sim
    import joystick_panel

    # Keep the run deterministic and headless.
    rov_sim.SLEEP_REALTIME = False
    rov_sim.HUD_ENABLED = False
    rov_sim.ENABLE_MARKERS = False
    rov_sim.ENABLE_CAMERA_PREVIEW = False
    rov_sim.ENABLE_THRUSTER_ARROWS = False

    sim_data = _simulate_control_loop(rov_sim, joystick_panel)
    write_json(ctx.module_dir / "trace.json", sim_data)

    results: list[CheckResult] = []

    results.append(
        CheckResult(
            name="shared_memory_to_mixer_trace_present",
            status="pass" if sim_data["trace"] else "fail",
            summary="Recorded deterministic trace containing shared state, mixer outputs, and thruster states.",
            evidence={"samples": len(sim_data["trace"]), "thruster_count": sim_data["thruster_count"]},
        )
    )

    has_yaw = any(sample["phase"].startswith("yaw") and abs(sample["net_force_world"][0]) > 0.001 for sample in sim_data["trace"])
    results.append(
        CheckResult(
            name="yaw_phase_generates_force",
            status="pass" if has_yaw else "warn",
            summary="Checked that yaw phases generate non-trivial world-frame force signatures.",
            evidence={"has_yaw_force_signature": has_yaw},
        )
    )

    results.append(
        CheckResult(
            name="cooldown_blocks_detected",
            status="pass" if sim_data["cooldown_block_count"] > 0 else "warn",
            summary="Verified reverse cooldown produced at least one blocked sign flip in deterministic run.",
            evidence={
                "cooldown_block_count": sim_data["cooldown_block_count"],
                "accepted_flip_count": sim_data["accepted_flip_count"],
                "cooldown_seconds": rov_sim.JOYSTICK_SWITCH_COOLDOWN,
            },
        )
    )

    payload = emit_module_results(ctx, "control_path", results, extras={"final_position": sim_data["final_position"]})
    console(f"[control_path] wrote: {ctx.module_dir / 'results.json'}", quiet=args.quiet)
    console(f"[control_path] checks: {payload['counts']}", quiet=args.quiet)
    return 1 if payload["counts"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

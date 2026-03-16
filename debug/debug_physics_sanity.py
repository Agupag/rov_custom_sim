#!/usr/bin/env python3
"""Physics sanity checks for signs, drift, and qualitative behavior."""

from __future__ import annotations

import math

import pybullet as p
import pybullet_data

from debug.common import CheckResult, console, emit_module_results, init_context, parse_common_args, write_json


def _vmag(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _step_world(rov_sim, rov, thrusters, thr_cmd, thr_level):
    base_pos, base_quat = p.getBasePositionAndOrientation(rov)
    lin, ang = p.getBaseVelocity(rov)

    depth = max(0.0, rov_sim.SURFACE_Z - base_pos[2])
    hull_half_z = 0.15
    submersion = depth / hull_half_z if depth < hull_half_z else 1.0

    depth_buoyancy_factor = max(0.5, 1.0 - rov_sim.DEPTH_BUOYANCY_COMPRESSIBILITY * depth)
    buoy_force = rov_sim.MASS * rov_sim.GRAVITY * rov_sim.BUOYANCY_SCALE * depth_buoyancy_factor * submersion

    cob_rel_world = p.rotateVector(base_quat, rov_sim.COB_OFFSET_BODY)
    cob_world = (
        base_pos[0] + cob_rel_world[0],
        base_pos[1] + cob_rel_world[1],
        base_pos[2] + cob_rel_world[2],
    )
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
    lin2, ang2 = p.getBaseVelocity(rov)
    pos2, quat2 = p.getBasePositionAndOrientation(rov)

    return {
        "pos": pos2,
        "quat": quat2,
        "lin": lin2,
        "ang": ang2,
        "speed": _vmag(lin2),
        "net_force": tuple(net_force),
        "submersion": submersion,
    }


def _run_scenario(rov_sim, thrusters, rov, seconds, cmd):
    steps = int(round(seconds / rov_sim.DT))
    thr_level = [0.0] * len(thrusters)
    samples = []
    for _ in range(steps):
        out = _step_world(rov_sim, rov, thrusters, cmd, thr_level)
        samples.append(out)
    return samples


def main() -> int:
    args = parse_common_args("Run physics sanity and sign checks")
    ctx = init_context("physics", args)

    import rov_sim
    from joystick_panel import mix_joystick_to_thruster_cmds

    rov_sim.SLEEP_REALTIME = False
    rov_sim.HUD_ENABLED = False
    rov_sim.ENABLE_MARKERS = False
    rov_sim.ENABLE_CAMERA_PREVIEW = False
    rov_sim.ENABLE_THRUSTER_ARROWS = False

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -rov_sim.GRAVITY)
    p.setPhysicsEngineParameter(fixedTimeStep=rov_sim.DT, numSolverIterations=50, numSubSteps=1)
    p.loadURDF("plane.urdf", [0, 0, rov_sim.SEABED_Z])

    rov, mesh_center = rov_sim.build_rov()
    auto_thr = rov_sim.detect_thrusters_from_gltf(rov_sim.GLTF_FILE, mesh_center)
    thrusters = auto_thr if auto_thr else rov_sim.THRUSTERS
    n_thr = len(thrusters)

    base_orn = p.getQuaternionFromEuler([math.radians(x) for x in rov_sim.MESH_BODY_EULER_DEG])

    def reset(depth=1.0):
        p.resetBasePositionAndOrientation(rov, [0, 0, rov_sim.SURFACE_Z - depth], base_orn)
        p.resetBaseVelocity(rov, [0, 0, 0], [0, 0, 0])
        rov_sim.LAST_VREL_BODY = None
        rov_sim.LAST_A_BODY = (0.0, 0.0, 0.0)
        rov_sim.LAST_W_BODY = None
        rov_sim.LAST_ALPHA_BODY = (0.0, 0.0, 0.0)

    # settle drift check
    reset(depth=1.0)
    settle = _run_scenario(rov_sim, thrusters, rov, 4.0, [0.0] * n_thr)

    # pure surge
    reset(depth=1.0)
    surge_cmd = mix_joystick_to_thruster_cmds({"surge": 1.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}, n_thr)
    surge = _run_scenario(rov_sim, thrusters, rov, 3.0, surge_cmd)

    # pure yaw
    reset(depth=1.0)
    yaw_cmd = mix_joystick_to_thruster_cmds({"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 1.0}, n_thr)
    yaw = _run_scenario(rov_sim, thrusters, rov, 3.0, yaw_cmd)

    # pure heave
    reset(depth=1.0)
    heave_cmd = [0.0] * n_thr
    if n_thr > 2:
        heave_cmd[2] = 1.0
    heave = _run_scenario(rov_sim, thrusters, rov, 3.0, heave_cmd)

    p.disconnect(cid)

    def mean_speed(samples):
        if not samples:
            return 0.0
        return sum(s["speed"] for s in samples) / len(samples)

    trace = {
        "settle": {
            "mean_speed": mean_speed(settle),
            "final_depth": rov_sim.SURFACE_Z - settle[-1]["pos"][2] if settle else None,
        },
        "surge": {
            "max_speed": max((s["speed"] for s in surge), default=0.0),
            "final_x": surge[-1]["pos"][0] if surge else None,
        },
        "yaw": {
            "max_abs_yaw_rate": max((abs(s["ang"][2]) for s in yaw), default=0.0),
            "final_yaw_rate": yaw[-1]["ang"][2] if yaw else None,
        },
        "heave": {
            "max_abs_vz": max((abs(s["lin"][2]) for s in heave), default=0.0),
            "final_z": heave[-1]["pos"][2] if heave else None,
        },
    }
    write_json(ctx.module_dir / "trace.json", trace)

    results: list[CheckResult] = []
    results.append(
        CheckResult(
            name="zero_input_settle_stability",
            status="pass" if trace["settle"]["mean_speed"] < 0.15 else "warn",
            summary="Checked for low drift under zero-input settle scenario.",
            evidence=trace["settle"],
        )
    )
    results.append(
        CheckResult(
            name="surge_command_moves_vehicle",
            status="pass" if (trace["surge"]["final_x"] or 0.0) > 0.03 else "fail",
            summary="Checked that positive surge command produces forward displacement.",
            evidence=trace["surge"],
        )
    )
    results.append(
        CheckResult(
            name="yaw_command_generates_yaw_rate",
            status="pass" if trace["yaw"]["max_abs_yaw_rate"] > 0.05 else "fail",
            summary="Checked that yaw command generates non-trivial angular z rate.",
            evidence=trace["yaw"],
        )
    )
    results.append(
        CheckResult(
            name="heave_command_generates_vertical_motion",
            status="pass" if trace["heave"]["max_abs_vz"] > 0.03 else "warn",
            summary="Checked that heave command creates vertical velocity response.",
            evidence=trace["heave"],
        )
    )

    payload = emit_module_results(ctx, "physics", results)
    console(f"[physics] wrote: {ctx.module_dir / 'results.json'}", quiet=args.quiet)
    console(f"[physics] checks: {payload['counts']}", quiet=args.quiet)
    return 1 if payload["counts"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

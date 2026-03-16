#!/usr/bin/env python3
"""One-factor-at-a-time sensitivity sweep for key ROV physics parameters.

This tool runs headless PyBullet scenarios and reports how small parameter changes
move the main behavior metrics used in diagnostics:
  - Surge terminal speed
  - Stopping distance / stopping time
  - Yaw steady-state rate
  - Heave speed

It is intentionally conservative: each sweep changes exactly one parameter while
holding all other constants at baseline values.
"""

import csv
import json
import math
import os
import sys
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pybullet as p
import pybullet_data

import rov_sim
from joystick_panel import mix_joystick_to_thruster_cmds


def vmag(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@contextmanager
def patched_rov_constants(patches):
    """Temporarily patch module-level constants on rov_sim."""
    original = {}
    for key, value in patches.items():
        original[key] = getattr(rov_sim, key)
        setattr(rov_sim, key, value)
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(rov_sim, key, value)


def setup_world():
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -rov_sim.GRAVITY)
    p.setPhysicsEngineParameter(fixedTimeStep=rov_sim.DT, numSolverIterations=50, numSubSteps=1)
    p.loadURDF("plane.urdf", [0, 0, rov_sim.SEABED_Z])

    rov, mesh_center = rov_sim.build_rov()
    auto_thr = rov_sim.detect_thrusters_from_gltf(rov_sim.GLTF_FILE, mesh_center)
    thrusters = auto_thr if auto_thr else rov_sim.THRUSTERS

    base_orn = p.getQuaternionFromEuler([math.radians(x) for x in rov_sim.MESH_BODY_EULER_DEG])
    return cid, rov, thrusters, base_orn


def reset_rov(rov, base_orn, z=0.4):
    p.resetBasePositionAndOrientation(rov, [0, 0, z], base_orn)
    p.resetBaseVelocity(rov, [0, 0, 0], [0, 0, 0])
    rov_sim.LAST_VREL_BODY = None
    rov_sim.LAST_A_BODY = (0.0, 0.0, 0.0)
    rov_sim.LAST_W_BODY = None
    rov_sim.LAST_ALPHA_BODY = (0.0, 0.0, 0.0)


def apply_step(rov, thrusters, thr_levels, thr_cmd):
    base_pos, base_quat = p.getBasePositionAndOrientation(rov)
    lin, ang = p.getBaseVelocity(rov)

    depth = max(0.0, rov_sim.SURFACE_Z - base_pos[2])
    hull_half_z = 0.15
    if depth < hull_half_z:
        submersion = min(1.0, max(0.0, depth / hull_half_z))
    else:
        submersion = 1.0

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
        rov_sim.apply_righting_torque(rov, base_quat, ang, submersion)
    rov_sim.apply_drag(rov, base_pos, base_quat, lin, ang)

    inv_q = p.invertTransform([0, 0, 0], base_quat)[1]
    v_body = p.rotateVector(inv_q, lin)

    for i, thr in enumerate(thrusters):
        cmd = thr_cmd[i] if i < len(thr_cmd) else 0.0
        tau = rov_sim.THRUSTER_TAU_UP if cmd > thr_levels[i] else rov_sim.THRUSTER_TAU_DN
        thr_levels[i] += (rov_sim.DT / max(1e-6, tau)) * (cmd - thr_levels[i])
        thr_levels[i] = max(-1.0, min(1.0, thr_levels[i]))

        if abs(thr_levels[i]) <= 1e-4:
            continue

        thrust_max = rov_sim.MAX_THRUST_H if thr["kind"] == "H" else rov_sim.MAX_THRUST_V
        thrust = thrust_max * thr_levels[i] * rov_sim.THRUST_LEVEL
        if thrust < 0.0:
            thrust *= rov_sim.BACKWARDS_THRUST_SCALE

        dir_body = thr.get("dir", (1.0, 0.0, 0.0))
        speed_along = v_body[0] * dir_body[0] + v_body[1] * dir_body[1] + v_body[2] * dir_body[2]
        loss = rov_sim.THRUSTER_SPEED_LOSS_COEF * abs(speed_along)
        loss = max(0.0, min(0.9, loss))
        thrust *= (1.0 - loss)

        force_body = (dir_body[0] * thrust, dir_body[1] * thrust, dir_body[2] * thrust)
        force_world = p.rotateVector(base_quat, force_body)
        pos_world = p.rotateVector(base_quat, thr["pos"])
        p.applyExternalForce(rov, -1, force_world, pos_world, p.WORLD_FRAME)

    p.stepSimulation()


def run_segment(rov, thrusters, thr_cmd, seconds):
    steps = int(seconds / rov_sim.DT)
    thr_levels = [0.0] * len(thrusters)
    samples = []
    for step in range(steps):
        t = step * rov_sim.DT
        apply_step(rov, thrusters, thr_levels, thr_cmd)
        if step % max(1, int(0.05 / rov_sim.DT)) == 0:
            pos, quat = p.getBasePositionAndOrientation(rov)
            lin, _ = p.getBaseVelocity(rov)
            yaw = math.degrees(p.getEulerFromQuaternion(quat)[2])
            samples.append({"t": t, "pos": pos, "lin": lin, "speed": vmag(lin), "yaw": yaw})
    return samples


def metrics_for_current_constants():
    cid, rov, thrusters, base_orn = setup_world()
    n_thr = len(thrusters)

    try:
        fwd_cmd = mix_joystick_to_thruster_cmds({"surge": 1.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}, n_thr)
        yaw_cmd = mix_joystick_to_thruster_cmds({"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 1.0}, n_thr)
        heave_cmd = mix_joystick_to_thruster_cmds({"surge": 0.0, "sway": 0.0, "heave": 1.0, "yaw": 0.0}, n_thr)

        reset_rov(rov, base_orn)
        run_segment(rov, thrusters, [0.0] * n_thr, 2.0)
        surge_data = run_segment(rov, thrusters, fwd_cmd, 16.0)

        surge_max = max((d["speed"] for d in surge_data), default=0.0)

        coast_start_pos = surge_data[-1]["pos"] if surge_data else (0.0, 0.0, 0.0)
        coast_start_speed = surge_data[-1]["speed"] if surge_data else 0.0
        stop_data = run_segment(rov, thrusters, [0.0] * n_thr, 8.0)
        coast_end_pos = stop_data[-1]["pos"] if stop_data else coast_start_pos
        dx = coast_end_pos[0] - coast_start_pos[0]
        dy = coast_end_pos[1] - coast_start_pos[1]
        stop_dist = math.sqrt(dx * dx + dy * dy)

        t_stop_10 = None
        stop_target = 0.1 * coast_start_speed
        for d in stop_data:
            if d["speed"] <= stop_target:
                t_stop_10 = d["t"]
                break

        reset_rov(rov, base_orn)
        run_segment(rov, thrusters, [0.0] * n_thr, 2.0)
        yaw_data = run_segment(rov, thrusters, yaw_cmd, 10.0)
        yaw_rates = []
        for i in range(1, len(yaw_data)):
            dt = yaw_data[i]["t"] - yaw_data[i - 1]["t"]
            if dt <= 0:
                continue
            dyaw = yaw_data[i]["yaw"] - yaw_data[i - 1]["yaw"]
            if dyaw > 180:
                dyaw -= 360
            if dyaw < -180:
                dyaw += 360
            yaw_rates.append(abs(dyaw / dt))
        yaw_steady = sum(yaw_rates[-20:]) / max(1, len(yaw_rates[-20:])) if yaw_rates else 0.0

        reset_rov(rov, base_orn)
        run_segment(rov, thrusters, [0.0] * n_thr, 2.0)
        heave_data = run_segment(rov, thrusters, heave_cmd, 8.0)
        heave_max = max((d["lin"][2] for d in heave_data), default=0.0)

        return {
            "surge_max_speed_mps": surge_max,
            "stop_distance_m": stop_dist,
            "stop_time_10pct_s": t_stop_10,
            "yaw_steady_rate_deg_s": yaw_steady,
            "heave_max_speed_mps": heave_max,
        }
    finally:
        p.disconnect(cid)


def build_scenarios():
    base_lin = tuple(rov_sim.LIN_DRAG_BODY)
    base_added = tuple(rov_sim.ADDED_MASS_BODY)
    base_loss = float(rov_sim.THRUSTER_SPEED_LOSS_COEF)

    scenarios = [
        {
            "name": "baseline",
            "patches": {},
            "notes": "Current constants unchanged",
        }
    ]

    for mult in (0.8, 1.2):
        scenarios.append(
            {
                "name": f"lin_drag_body_x_{mult:.1f}x",
                "patches": {
                    "LIN_DRAG_BODY": (base_lin[0] * mult, base_lin[1], base_lin[2]),
                },
                "notes": "Sweep surge linear drag only",
            }
        )

    for mult in (0.8, 1.2):
        scenarios.append(
            {
                "name": f"added_mass_body_x_{mult:.1f}x",
                "patches": {
                    "ADDED_MASS_BODY": (base_added[0] * mult, base_added[1], base_added[2]),
                },
                "notes": "Sweep surge added mass only",
            }
        )

    delta = 0.03
    scenarios.append(
        {
            "name": "thruster_speed_loss_minus_0p03",
            "patches": {"THRUSTER_SPEED_LOSS_COEF": max(0.0, base_loss - delta)},
            "notes": "Lower speed-dependent thrust loss",
        }
    )
    scenarios.append(
        {
            "name": "thruster_speed_loss_plus_0p03",
            "patches": {"THRUSTER_SPEED_LOSS_COEF": base_loss + delta},
            "notes": "Higher speed-dependent thrust loss",
        }
    )

    return scenarios


def run_sweep():
    out_dir = os.path.join(ROOT, "tools")
    os.makedirs(out_dir, exist_ok=True)

    scenarios = build_scenarios()
    rows = []

    print("=" * 72)
    print("SENSITIVITY SWEEP (one-factor-at-a-time)")
    print("=" * 72)

    for i, scenario in enumerate(scenarios, start=1):
        print(f"[{i}/{len(scenarios)}] {scenario['name']}")
        with patched_rov_constants(scenario["patches"]):
            metrics = metrics_for_current_constants()
        row = {
            "scenario": scenario["name"],
            "notes": scenario["notes"],
            "patches": scenario["patches"],
            **metrics,
        }
        rows.append(row)

    json_path = os.path.join(out_dir, "sensitivity_sweep_results.json")
    csv_path = os.path.join(out_dir, "sensitivity_sweep_results.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "surge_max_speed_mps",
                "stop_distance_m",
                "stop_time_10pct_s",
                "yaw_steady_rate_deg_s",
                "heave_max_speed_mps",
                "patches",
                "notes",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["scenario"],
                    f"{row['surge_max_speed_mps']:.4f}",
                    f"{row['stop_distance_m']:.4f}",
                    "" if row["stop_time_10pct_s"] is None else f"{row['stop_time_10pct_s']:.4f}",
                    f"{row['yaw_steady_rate_deg_s']:.4f}",
                    f"{row['heave_max_speed_mps']:.4f}",
                    json.dumps(row["patches"]),
                    row["notes"],
                ]
            )

    print("\nSaved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    # Keep diagnostics deterministic and fast.
    rov_sim.SLEEP_REALTIME = False
    rov_sim.HUD_ENABLED = False
    rov_sim.ENABLE_MARKERS = False
    rov_sim.ENABLE_CAMERA_PREVIEW = False
    rov_sim.LOG_PHYSICS_DETAILED = False
    rov_sim.ENABLE_THRUSTER_ARROWS = False

    run_sweep()

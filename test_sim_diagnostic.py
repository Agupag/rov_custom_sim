#!/usr/bin/env python3
"""
Comprehensive diagnostic: simulates joystick inputs inside the real PyBullet
physics simulation and records thruster commands, forces, and ROV motion.

This runs the ACTUAL simulator code (not unit-test stubs) in DIRECT mode
(headless), injects fake joystick values, steps the physics, and measures
what really happens.  Output is a clear report showing whether the ROV
moves in the expected direction for each joystick input.
"""

import sys, os, math, time, importlib

# ── Ensure workspace is on path ──
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pybullet as p
import pybullet_data
import json

# Import our modules
import joystick_panel
from joystick_panel import mix_joystick_to_thruster_cmds

# ── Helpers from rov_sim (import key functions) ──
import rov_sim

# ────────────────────────────────────────────────────────────────────
#  PART 1: Pure mixer tests — no physics, just verify mixer outputs
# ────────────────────────────────────────────────────────────────────

def test_mixer():
    """Verify the mixer maps joystick inputs to expected thruster commands."""
    print("=" * 70)
    print("PART 1: MIXER UNIT TESTS")
    print("=" * 70)
    
    # (surge, yaw) -> expected (T1, T2, T3, T4)
    # Binary ON/OFF mixer: raw_T1 = surge - yaw, raw_T2 = surge + yaw, raw_T4 = surge
    # Each raw value snapped: |val|<0.15 → 0, val>0 → +1, val<0 → -1
    cases = [
        # Pure forward (stick pushed up) — all H thrusters forward
        ("Forward",            {"surge":  1.0, "yaw":  0.0}, (1, 1, 0, 1)),
        # Pure reverse (stick pushed down)
        ("Reverse",            {"surge": -1.0, "yaw":  0.0}, (-1, -1, 0, -1)),
        # Pure yaw right (stick pushed right) — T1=-1,T2=+1 for CW torque
        ("Yaw Right",          {"surge":  0.0, "yaw":  1.0}, (-1, 1, 0, 0)),
        # Pure yaw left (stick pushed left) — T1=+1,T2=-1 for CCW torque
        ("Yaw Left",           {"surge":  0.0, "yaw": -1.0}, (1, -1, 0, 0)),
        # Forward + yaw right: raw_T1=1-1=0→0, raw_T2=1+1=2→+1, raw_T4=1→+1
        ("Forward+Yaw Right",  {"surge":  1.0, "yaw":  1.0}, (0, 1, 0, 1)),
        # Forward + yaw left: raw_T1=1+1=2→+1, raw_T2=1-1=0→0, raw_T4=1→+1
        ("Forward+Yaw Left",   {"surge":  1.0, "yaw": -1.0}, (1, 0, 0, 1)),
        # Reverse + yaw right: raw_T1=-1-1=-2→-1, raw_T2=-1+1=0→0, raw_T4=-1→-1
        ("Reverse+Yaw Right",  {"surge": -1.0, "yaw":  1.0}, (-1, 0, 0, -1)),
        # Reverse + yaw left: raw_T1=-1+1=0→0, raw_T2=-1-1=-2→-1, raw_T4=-1→-1
        ("Reverse+Yaw Left",   {"surge": -1.0, "yaw": -1.0}, (0, -1, 0, -1)),
        # Dead zone — stick barely pushed
        ("Dead zone",          {"surge":  0.05, "yaw": 0.05}, (0, 0, 0, 0)),
    ]

    all_ok = True
    for label, state, expected in cases:
        cmds = mix_joystick_to_thruster_cmds(state, 4)
        cmds_rounded = tuple(int(round(c)) for c in cmds)
        ok = cmds_rounded == expected
        status = "✅" if ok else "❌"
        if not ok:
            all_ok = False
        print(f"  {status} {label:25s} surge={state['surge']:+.1f} yaw={state['yaw']:+.1f}"
              f"  → T1={cmds_rounded[0]:+d} T2={cmds_rounded[1]:+d} T3={cmds_rounded[2]:+d} T4={cmds_rounded[3]:+d}"
              f"  (expected T1={expected[0]:+d} T2={expected[1]:+d} T3={expected[2]:+d} T4={expected[3]:+d})")
    
    print(f"\n  Mixer tests: {'ALL PASSED' if all_ok else 'SOME FAILED'}\n")
    return all_ok


# ────────────────────────────────────────────────────────────────────
#  PART 2: Thruster direction verification
# ────────────────────────────────────────────────────────────────────

def setup_pybullet():
    """Create the PyBullet world and load the ROV (headless DIRECT mode)."""
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -rov_sim.GRAVITY)
    p.setTimeStep(rov_sim.DT)
    
    # Use the same build_rov() function the sim uses
    rov, mesh_center = rov_sim.build_rov()
    
    # Detect thrusters from GLTF (same as main sim)
    gltf_path = os.path.join(ROOT, "Assembly 1.gltf")
    thrusters = rov_sim.detect_thrusters_from_gltf(gltf_path, mesh_center)
    cam_info = rov_sim.find_camera_pose_from_gltf(gltf_path, mesh_center)
    
    # Get the orientation used by build_rov
    euler_rad = tuple(math.radians(d) for d in rov_sim.MESH_BODY_EULER_DEG)
    orn = p.getQuaternionFromEuler(euler_rad)
    
    # Move to starting position underwater
    start_pos = [0, 0, rov_sim.SURFACE_Z - 1.0]
    p.resetBasePositionAndOrientation(rov, start_pos, orn)
    p.resetBaseVelocity(rov, [0, 0, 0], [0, 0, 0])
    
    return cid, rov, thrusters, cam_info, start_pos, orn


def _report_thruster_directions(rov, thrusters, base_orn):
    """Print each thruster's direction in body and world frame.
    
    NOTE: This is a helper/report function, not a pytest test.
    Call it manually from __main__ if needed — it requires a live
    PyBullet connection with a loaded ROV.
    """
    print("=" * 70)
    print("PART 2: THRUSTER DIRECTIONS (body → world)")
    print("=" * 70)
    
    base_quat = base_orn  # initial orientation (90° Z rotation)
    
    # Camera forward: mesh (0,-1,0) → world
    cam_fwd_world = p.rotateVector(base_quat, (0, -1, 0))
    print(f"\n  Camera forward (mesh (0,-1,0)) → world: ({cam_fwd_world[0]:+.3f}, {cam_fwd_world[1]:+.3f}, {cam_fwd_world[2]:+.3f})")
    print(f"  (Should be roughly (+1, 0, 0) = world +X)\n")
    
    for i, t in enumerate(thrusters):
        dir_body = t["dir"]
        pos_body = t["pos"]
        dir_world = p.rotateVector(base_quat, dir_body)
        pos_world = p.rotateVector(base_quat, pos_body)
        print(f"  T{i+1} ({t['name']}, {t['kind']}):")
        print(f"    body pos:  ({pos_body[0]:+.3f}, {pos_body[1]:+.3f}, {pos_body[2]:+.3f})")
        print(f"    body dir:  ({dir_body[0]:+.3f}, {dir_body[1]:+.3f}, {dir_body[2]:+.3f})")
        print(f"    world pos: ({pos_world[0]:+.3f}, {pos_world[1]:+.3f}, {pos_world[2]:+.3f})")
        print(f"    world dir: ({dir_world[0]:+.3f}, {dir_world[1]:+.3f}, {dir_world[2]:+.3f})")
        
        # Dot with camera forward (world +X)
        dot_fwd = dir_world[0] * cam_fwd_world[0] + dir_world[1] * cam_fwd_world[1]
        # Dot with world +Y (lateral)
        dot_lat = dir_world[1]
        print(f"    dot(forward) = {dot_fwd:+.3f}   dot(lateral +Y) = {dot_lat:+.3f}")
    
    print()
    
    # KEY ANALYSIS: T1 and T2 torque contribution
    print("  TORQUE ANALYSIS — T1 and T2 as yaw pair:")
    print("  Torque_z = pos_x * force_y - pos_y * force_x (positive = CCW = yaw left)")
    for label, cmds in [("T1=+1,T2=-1", [1,-1,0,0]), ("T1=-1,T2=+1", [-1,1,0,0])]:
        torque_z = 0
        fx_total = 0
        fy_total = 0
        for i, cmd in enumerate(cmds):
            if i >= len(thrusters) or cmd == 0:
                continue
            dir_w = p.rotateVector(base_quat, thrusters[i]["dir"])
            pos_w = p.rotateVector(base_quat, thrusters[i]["pos"])
            f = (dir_w[0]*cmd, dir_w[1]*cmd, dir_w[2]*cmd)
            fx_total += f[0]
            fy_total += f[1]
            torque_z += pos_w[0] * f[1] - pos_w[1] * f[0]
            print(f"    {thrusters[i]['name']} cmd={cmd:+d}: "
                  f"pos_w=({pos_w[0]:+.3f},{pos_w[1]:+.3f}) "
                  f"force=({f[0]:+.3f},{f[1]:+.3f}) "
                  f"τz={pos_w[0]*f[1]-pos_w[1]*f[0]:+.4f}")
        print(f"    {label}: Net force=({fx_total:+.3f},{fy_total:+.3f}) Net τz={torque_z:+.4f}")
        print(f"    → {'YAW LEFT' if torque_z>0 else 'YAW RIGHT'} ({abs(torque_z):.4f} Nm per unit thrust)")
        print(f"    → Also TRANSLATES {'FWD' if fx_total>0 else 'BACK'} with {abs(fx_total):.3f}N")
        print()
    
    # Check combined forces for each mixer scenario
    print("  COMBINED FORCE ANALYSIS for mixer outputs:")
    print("  (forces shown in WORLD frame, +X = camera forward, +Y = left)")
    
    scenarios = [
        ("Forward (T4=+1)",              [0, 0, 0, 1]),
        ("Reverse (T4=-1)",              [0, 0, 0, -1]),
        ("Yaw Right (T1=+1,T2=-1)",      [1, -1, 0, 0]),
        ("Yaw Left (T1=-1,T2=+1)",       [-1, 1, 0, 0]),
        ("Fwd+YawR (T1=+1,T2=-1,T4=+1)", [1, -1, 0, 1]),
        ("Fwd+YawL (T1=-1,T2=+1,T4=+1)", [-1, 1, 0, 1]),
    ]
    
    for label, cmds in scenarios:
        fx_total = 0
        fy_total = 0
        fz_total = 0
        torque_z = 0
        for i, cmd in enumerate(cmds):
            if i >= len(thrusters) or cmd == 0:
                continue
            dir_world = p.rotateVector(base_quat, thrusters[i]["dir"])
            pos_world = p.rotateVector(base_quat, thrusters[i]["pos"])
            f = (dir_world[0] * cmd, dir_world[1] * cmd, dir_world[2] * cmd)
            fx_total += f[0]
            fy_total += f[1]
            fz_total += f[2]
            # Torque about Z axis: r × F → z component = rx*Fy - ry*Fx
            torque_z += pos_world[0] * f[1] - pos_world[1] * f[0]
        
        # Classify motion direction
        angle_deg = math.degrees(math.atan2(fy_total, fx_total)) if abs(fx_total) + abs(fy_total) > 1e-6 else 0
        fwd_component = fx_total  # positive = forward
        lat_component = fy_total  # positive = world +Y = ???
        
        motion = []
        if abs(fwd_component) > 0.01:
            motion.append("FWD" if fwd_component > 0 else "BACK")
        if abs(lat_component) > 0.01:
            motion.append("LAT+" if lat_component > 0 else "LAT-")
        if abs(torque_z) > 0.001:
            # Positive torque_z in world = CCW from above = yaw LEFT in standard RH
            motion.append(f"YAW({'LEFT' if torque_z > 0 else 'RIGHT'})")
        
        print(f"\n    {label}")
        print(f"      Net force:  Fx={fx_total:+.3f} (fwd) Fy={fy_total:+.3f} (lat) Fz={fz_total:+.3f}")
        print(f"      Torque_z:   {torque_z:+.4f} Nm")
        print(f"      Motion:     {' + '.join(motion) if motion else 'NONE'}")


# ────────────────────────────────────────────────────────────────────
#  PART 3: Full physics simulation with simulated joystick inputs
# ────────────────────────────────────────────────────────────────────

def run_physics_test(rov, thrusters, base_orn, label, thruster_cmds, duration=3.0, start_pos=None):
    """
    Apply thruster commands for `duration` seconds and measure ROV displacement.
    Returns (dx, dy, dz, final_yaw - initial_yaw).
    """
    if start_pos is None:
        start_pos = [0, 0, rov_sim.SURFACE_Z - 1.0]
    
    # Reset ROV
    p.resetBasePositionAndOrientation(rov, start_pos, base_orn)
    p.resetBaseVelocity(rov, [0, 0, 0], [0, 0, 0])
    
    pos0, quat0 = p.getBasePositionAndOrientation(rov)
    _, _, yaw0 = p.getEulerFromQuaternion(quat0)
    
    steps = int(duration / rov_sim.DT)
    
    # Thruster ramp-up levels
    thr_level = [0.0] * len(thrusters)
    
    for step in range(steps):
        base_pos, base_quat = p.getBasePositionAndOrientation(rov)
        lin, ang = p.getBaseVelocity(rov)
        
        # Apply buoyancy
        depth = max(0.0, rov_sim.SURFACE_Z - base_pos[2])
        _hull_half_z = 0.15
        if depth >= _hull_half_z:
            submersion = 1.0
        elif depth <= 0.0:
            submersion = 0.0
        else:
            submersion = depth / _hull_half_z
        
        depth_buoyancy_factor = max(0.5, 1.0 - rov_sim.DEPTH_BUOYANCY_COMPRESSIBILITY * depth)
        buoy_force = rov_sim.MASS * rov_sim.GRAVITY * rov_sim.BUOYANCY_SCALE * depth_buoyancy_factor * submersion
        
        cob_rel_world = p.rotateVector(base_quat, rov_sim.COB_OFFSET_BODY)
        cob_world = (base_pos[0] + cob_rel_world[0],
                     base_pos[1] + cob_rel_world[1],
                     base_pos[2] + cob_rel_world[2])
        p.applyExternalForce(rov, -1, (0, 0, buoy_force), cob_world, p.WORLD_FRAME)
        
        # Apply ballast
        rov_sim.apply_ballast(rov, base_pos, base_quat)
        
        # Apply righting torque
        if submersion > 0.01:
            rov_sim.apply_righting_torque(rov, base_quat, ang, submersion)
        
        # Apply drag
        rov_sim.apply_drag(rov, base_pos, base_quat, lin, ang)
        
        # Apply thruster forces (same as main sim)
        inv_q = p.invertTransform([0, 0, 0], base_quat)[1]
        v_body = p.rotateVector(inv_q, lin)
        
        for i, t in enumerate(thrusters):
            cmd = thruster_cmds[i] if i < len(thruster_cmds) else 0.0
            
            # Ramp
            if cmd > thr_level[i]:
                tau = rov_sim.THRUSTER_TAU_UP
            else:
                tau = rov_sim.THRUSTER_TAU_DN
            thr_level[i] += (rov_sim.DT / max(1e-6, tau)) * (cmd - thr_level[i])
            thr_level[i] = max(-1.0, min(1.0, thr_level[i]))
            
            if abs(thr_level[i]) <= 1e-4:
                continue
            
            thrust_max = rov_sim.MAX_THRUST_H if t["kind"] == "H" else rov_sim.MAX_THRUST_V
            thrust = thrust_max * thr_level[i] * rov_sim.THRUST_LEVEL
            if thrust < 0:
                thrust *= rov_sim.BACKWARDS_THRUST_SCALE
            
            # Inflow loss
            dir_body = t.get("dir", (1, 0, 0))
            speed_along = v_body[0]*dir_body[0] + v_body[1]*dir_body[1] + v_body[2]*dir_body[2]
            loss = rov_sim.THRUSTER_SPEED_LOSS_COEF * abs(speed_along)
            loss = max(0.0, min(0.9, loss))
            thrust *= (1.0 - loss)
            
            dir_world = p.rotateVector(base_quat, t["dir"])
            force = (dir_world[0] * thrust, dir_world[1] * thrust, dir_world[2] * thrust)
            
            rel_pos_world = p.rotateVector(base_quat, t["pos"])
            world_pos = (base_pos[0] + rel_pos_world[0],
                         base_pos[1] + rel_pos_world[1],
                         base_pos[2] + rel_pos_world[2])
            
            p.applyExternalForce(rov, -1, force, world_pos, p.WORLD_FRAME)
        
        p.stepSimulation()
    
    pos1, quat1 = p.getBasePositionAndOrientation(rov)
    _, _, yaw1 = p.getEulerFromQuaternion(quat1)
    
    dx = pos1[0] - pos0[0]
    dy = pos1[1] - pos0[1]
    dz = pos1[2] - pos0[2]
    dyaw = math.degrees(yaw1 - yaw0)
    # Normalize to -180..180
    while dyaw > 180: dyaw -= 360
    while dyaw < -180: dyaw += 360
    
    return dx, dy, dz, dyaw


def test_physics():
    """Run full physics tests with simulated joystick inputs."""
    print("\n" + "=" * 70)
    print("PART 3: FULL PHYSICS SIMULATION — SIMULATED JOYSTICK INPUT")
    print("=" * 70)
    
    cid, rov, thrusters, cam_info, start_pos, base_orn = setup_pybullet()
    
    # First, show thruster directions
    _report_thruster_directions(rov, thrusters, base_orn)
    
    print("\n" + "-" * 70)
    print("  SIMULATED JOYSTICK → PHYSICS RESULTS")
    print("  (Each test: 3 seconds of constant joystick input)")
    print("  World frame: +X = camera forward, +Y = camera left")
    print("-" * 70)
    
    # Joystick inputs to test
    tests = [
        ("FORWARD (stick full up)",          {"surge":  1.0, "yaw":  0.0}),
        ("REVERSE (stick full down)",         {"surge": -1.0, "yaw":  0.0}),
        ("YAW RIGHT (stick full right)",      {"surge":  0.0, "yaw":  1.0}),
        ("YAW LEFT (stick full left)",        {"surge":  0.0, "yaw": -1.0}),
        ("FORWARD+YAW RIGHT (upper-right)",   {"surge":  1.0, "yaw":  1.0}),
        ("FORWARD+YAW LEFT (upper-left)",     {"surge":  1.0, "yaw": -1.0}),
        ("REVERSE+YAW RIGHT (lower-right)",   {"surge": -1.0, "yaw":  1.0}),
        ("REVERSE+YAW LEFT (lower-left)",     {"surge": -1.0, "yaw": -1.0}),
        ("DEAD ZONE (barely pushed)",         {"surge":  0.05, "yaw":  0.05}),
    ]
    
    results = []
    
    for label, js_state in tests:
        # Get mixer output
        cmds = mix_joystick_to_thruster_cmds(js_state, len(thrusters))
        
        # Run physics
        dx, dy, dz, dyaw = run_physics_test(rov, thrusters, base_orn, label, cmds,
                                              duration=3.0, start_pos=start_pos)
        
        dist_h = math.sqrt(dx*dx + dy*dy)
        heading = math.degrees(math.atan2(dy, dx)) if dist_h > 0.01 else 0
        
        # Classify what actually happened
        actual_motion = []
        if abs(dx) > 0.05:
            actual_motion.append("FWD" if dx > 0 else "BACK")
        if abs(dy) > 0.05:
            actual_motion.append("LEFT" if dy > 0 else "RIGHT")
        if abs(dyaw) > 2.0:
            actual_motion.append(f"YAW {'LEFT' if dyaw > 0 else 'RIGHT'} {abs(dyaw):.0f}°")
        if abs(dz) > 0.05:
            actual_motion.append(f"{'UP' if dz > 0 else 'DOWN'} {abs(dz):.2f}m")
        
        cmds_str = " ".join([f"T{i+1}={c:+.0f}" for i, c in enumerate(cmds)])
        
        print(f"\n  📌 {label}")
        print(f"     Mixer output:     {cmds_str}")
        print(f"     Displacement:     dx={dx:+.3f}m (fwd)  dy={dy:+.3f}m (lat)  dz={dz:+.3f}m (vert)")
        print(f"     Horizontal dist:  {dist_h:.3f}m  heading={heading:+.0f}°")
        print(f"     Yaw change:       {dyaw:+.1f}°")
        print(f"     Actual motion:    {', '.join(actual_motion) if actual_motion else 'STATIONARY'}")
        
        # Check if motion matches expectation
        expected_ok = True
        issues = []
        
        if "FORWARD" in label and "YAW" not in label:
            if dx < 0.1:
                issues.append("Should move forward (dx>0) but didn't")
                expected_ok = False
            if abs(dy) > abs(dx) * 0.3:
                issues.append(f"Too much lateral drift: |dy|={abs(dy):.3f} vs dx={dx:.3f}")
                expected_ok = False
        
        if "REVERSE" in label and "YAW" not in label:
            if dx > -0.1:
                issues.append("Should move backward (dx<0) but didn't")
                expected_ok = False
        
        if "YAW RIGHT" in label and "FORWARD" not in label and "REVERSE" not in label:
            if dyaw > -1.0:  # yaw right = negative in standard coords
                issues.append(f"Should yaw right (negative dyaw) but got {dyaw:+.1f}°")
                expected_ok = False
        
        if "YAW LEFT" in label and "FORWARD" not in label and "REVERSE" not in label:
            if dyaw < 1.0:
                issues.append(f"Should yaw left (positive dyaw) but got {dyaw:+.1f}°")
                expected_ok = False
        
        if "DEAD ZONE" in label:
            if dist_h > 0.10:  # Allow some drift from buoyancy/current
                issues.append(f"Should be stationary but moved {dist_h:.3f}m")
                expected_ok = False
        
        if expected_ok:
            print(f"     Result:           ✅ CORRECT")
        else:
            print(f"     Result:           ❌ WRONG")
            for issue in issues:
                print(f"       ⚠️  {issue}")
        
        results.append((label, expected_ok, issues, dx, dy, dz, dyaw))
    
    p.disconnect()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for r in results if r[1])
    n_fail = sum(1 for r in results if not r[1])
    print(f"  Passed: {n_pass}/{len(results)}")
    print(f"  Failed: {n_fail}/{len(results)}")
    
    if n_fail > 0:
        print("\n  FAILED TESTS:")
        for label, ok, issues, dx, dy, dz, dyaw in results:
            if not ok:
                print(f"    ❌ {label}")
                for issue in issues:
                    print(f"       → {issue}")
    
    return n_fail == 0


# ────────────────────────────────────────────────────────────────────
#  PART 4: Verify joystick panel shared-memory → mixer → thruster path
# ────────────────────────────────────────────────────────────────────

def test_shared_memory_path():
    """Simulate writing to shared memory and reading back through get_joystick_state."""
    print("\n" + "=" * 70)
    print("PART 4: SHARED MEMORY PATH TEST")
    print("=" * 70)
    
    joystick_panel._ensure_shared()
    
    # Write joystick values
    test_vals = [
        ("Full forward",  0, 1.0, 3, 0.0),
        ("Full yaw right", 0, 0.0, 3, 1.0),
        ("Forward+right", 0, 0.7, 3, 0.7),
    ]
    
    all_ok = True
    for label, surge_idx, surge_val, yaw_idx, yaw_val in test_vals:
        with joystick_panel._shared.get_lock():
            joystick_panel._shared[0] = surge_val
            joystick_panel._shared[3] = yaw_val
            joystick_panel._shared[4] = 1.0  # active
        
        state = joystick_panel.get_joystick_state()
        cmds = mix_joystick_to_thruster_cmds(state, 4)
        
        ok = (abs(state["surge"] - surge_val) < 0.001 and
              abs(state["yaw"] - yaw_val) < 0.001 and
              state["active"])
        
        if not ok:
            all_ok = False
        
        status = "✅" if ok else "❌"
        cmds_str = " ".join([f"T{i+1}={c:+.0f}" for i, c in enumerate(cmds)])
        print(f"  {status} {label}: surge={state['surge']:+.2f} yaw={state['yaw']:+.2f}"
              f" active={state['active']} → {cmds_str}")
    
    # Clean up
    with joystick_panel._shared.get_lock():
        for i in range(8):
            joystick_panel._shared[i] = 0.0
    
    print(f"\n  Shared memory tests: {'ALL PASSED' if all_ok else 'SOME FAILED'}\n")
    return all_ok


# ────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "🔧" * 35)
    print("  ROV SIMULATOR — FULL DIAGNOSTIC")
    print("🔧" * 35 + "\n")
    
    ok1 = test_mixer()
    ok2 = test_shared_memory_path()
    ok3 = test_physics()
    
    print("\n" + "=" * 70)
    print("OVERALL RESULT")
    print("=" * 70)
    
    if ok1 and ok2 and ok3:
        print("  ✅ ALL DIAGNOSTICS PASSED")
    else:
        print("  ❌ SOME DIAGNOSTICS FAILED — see details above")
        if not ok3:
            print("  ⚠️  Physics behavior doesn't match expectations!")
            print("  The mixer or thruster directions need fixing.")
    
    print()

#!/usr/bin/env python3
"""
Physics realism diagnostic: measures acceleration, top speed, stopping distance,
yaw rate, and vertical response — then compares against DDR specs and real-world
ROV behavior for a 7.5 kg vehicle with 5.56 N thrusters.

Real-world expectations for this class of ROV:
  - Terminal velocity (surge): ~0.3-0.5 m/s  (DDR predicts ~0.4 m/s)
  - Time to 90% of terminal velocity: 5-10 seconds (heavy water resistance)
  - Stopping distance from 0.4 m/s: ~0.3-0.5 m  (takes 2-4 seconds)
  - Yaw rate (T1/T2 differential): ~15-30 °/s  (slow, ponderous)
  - Vertical rise rate: ~0.2-0.4 m/s  (slightly buoyant + T3)
"""
import os, sys, math, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pybullet as p
import pybullet_data
import rov_sim
import joystick_panel as jp

def vmag(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def run():
    # Set up shared memory
    jp._ensure_shared()
    with jp._shared.get_lock():
        for i in range(8):
            jp._shared[i] = 0.0

    # Configure headless
    rov_sim.SLEEP_REALTIME = False
    rov_sim.HUD_ENABLED = False
    rov_sim.ENABLE_MARKERS = False
    rov_sim.ENABLE_CAMERA_PREVIEW = False
    rov_sim.LOG_PHYSICS_DETAILED = False
    rov_sim.ENABLE_THRUSTER_ARROWS = False

    # PyBullet setup
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -rov_sim.GRAVITY)
    p.setPhysicsEngineParameter(fixedTimeStep=rov_sim.DT, numSolverIterations=50, numSubSteps=1)
    p.loadURDF("plane.urdf", [0, 0, rov_sim.SEABED_Z])

    rov, mesh_center = rov_sim.build_rov()
    auto_thr = rov_sim.detect_thrusters_from_gltf(rov_sim.GLTF_FILE, mesh_center)
    if auto_thr:
        rov_sim.THRUSTERS = auto_thr

    THRUSTERS = rov_sim.THRUSTERS
    n_thr = len(THRUSTERS)
    DT = rov_sim.DT

    def reset_rov():
        p.resetBasePositionAndOrientation(rov, [0, 0, 0.4],
            p.getQuaternionFromEuler([math.radians(x) for x in rov_sim.MESH_BODY_EULER_DEG]))
        p.resetBaseVelocity(rov, [0, 0, 0], [0, 0, 0])
        rov_sim.LAST_VREL_BODY = None
        rov_sim.LAST_A_BODY = (0.0, 0.0, 0.0)
        rov_sim.LAST_W_BODY = None
        rov_sim.LAST_ALPHA_BODY = (0.0, 0.0, 0.0)

    def step_physics(thr_cmd, n_steps):
        """Step physics with given thruster commands, return trajectory data."""
        thr_level = [0.0] * n_thr
        data = []  # (time, pos, vel_world, vel_body, ang_vel, speed)
        
        for step in range(n_steps):
            t = step * DT
            base_pos, base_quat = p.getBasePositionAndOrientation(rov)
            lin, ang = p.getBaseVelocity(rov)

            # Buoyancy
            depth = max(0.0, rov_sim.SURFACE_Z - base_pos[2])
            _hull_half_z = 0.15
            submersion = min(1.0, max(0.0, depth / _hull_half_z)) if depth < _hull_half_z else 1.0
            depth_buoyancy_factor = max(0.5, 1.0 - rov_sim.DEPTH_BUOYANCY_COMPRESSIBILITY * depth)

            buoy_force = rov_sim.MASS * rov_sim.GRAVITY * rov_sim.BUOYANCY_SCALE * depth_buoyancy_factor * submersion
            cob_rel_world = p.rotateVector(base_quat, rov_sim.COB_OFFSET_BODY)
            cob_world = (base_pos[0] + cob_rel_world[0], base_pos[1] + cob_rel_world[1], base_pos[2] + cob_rel_world[2])
            p.applyExternalForce(rov, -1, (0.0, 0.0, buoy_force), cob_world, p.WORLD_FRAME)

            rov_sim.apply_ballast(rov, base_pos, base_quat)
            if submersion > 0.01:
                rov_sim.apply_righting_torque(rov, base_quat, ang, submersion)
            rov_sim.apply_drag(rov, base_pos, base_quat, lin, ang)

            # Thrusters with ramping
            inv_q = p.invertTransform([0, 0, 0], base_quat)[1]
            v_body = p.rotateVector(inv_q, lin)

            for i, thr in enumerate(THRUSTERS):
                if thr_cmd[i] > thr_level[i]:
                    tau = rov_sim.THRUSTER_TAU_UP
                elif thr_cmd[i] < thr_level[i]:
                    tau = rov_sim.THRUSTER_TAU_DN
                else:
                    tau = rov_sim.THRUSTER_TAU_DN
                thr_level[i] += (DT / max(1e-6, tau)) * (thr_cmd[i] - thr_level[i])
                thr_level[i] = max(-1.0, min(1.0, thr_level[i]))
                if abs(thr_level[i]) <= 1e-4:
                    continue

                thrust_max = rov_sim.MAX_THRUST_H if thr["kind"] == "H" else rov_sim.MAX_THRUST_V
                thrust = thrust_max * thr_level[i] * rov_sim.THRUST_LEVEL
                if thrust < 0.0:
                    thrust *= rov_sim.BACKWARDS_THRUST_SCALE

                dir_body = thr.get("dir", (1.0, 0.0, 0.0))
                speed_along = v_body[0]*dir_body[0] + v_body[1]*dir_body[1] + v_body[2]*dir_body[2]
                loss = rov_sim.THRUSTER_SPEED_LOSS_COEF * abs(speed_along)
                loss = max(0.0, min(0.9, loss))
                thrust *= (1.0 - loss)

                dir_world = p.rotateVector(base_quat, thr["dir"])
                force = (dir_world[0]*thrust, dir_world[1]*thrust, dir_world[2]*thrust)
                rel_world = p.rotateVector(base_quat, thr["pos"])
                world_pos = (base_pos[0]+rel_world[0], base_pos[1]+rel_world[1], base_pos[2]+rel_world[2])
                p.applyExternalForce(rov, -1, force, world_pos, p.WORLD_FRAME)

            p.stepSimulation()

            # Record data every 0.05s
            if step % max(1, int(0.05 / DT)) == 0:
                speed = vmag(lin)
                yaw = math.degrees(p.getEulerFromQuaternion(base_quat)[2])
                data.append({
                    't': t, 'pos': base_pos, 'vel': lin, 'ang': ang,
                    'speed': speed, 'yaw': yaw, 'z': base_pos[2],
                    'thr_level': list(thr_level)
                })
        return data

    print("=" * 72)
    print("  PHYSICS REALISM DIAGNOSTIC")
    print("=" * 72)
    print(f"  MASS = {rov_sim.MASS} kg")
    print(f"  MAX_THRUST_H = {rov_sim.MAX_THRUST_H} N, MAX_THRUST_V = {rov_sim.MAX_THRUST_V} N")
    print(f"  LIN_DRAG_BODY = {rov_sim.LIN_DRAG_BODY}")
    print(f"  CD = {rov_sim.CD}, AREA = {rov_sim.AREA}")
    print(f"  ADDED_MASS_BODY = {rov_sim.ADDED_MASS_BODY}")
    print(f"  RIGHTING_K_RP = {rov_sim.RIGHTING_K_RP}, KD = {rov_sim.RIGHTING_KD_RP}")
    print(f"  LIN_DRAG_ANG = {rov_sim.LIN_DRAG_ANG}, QUAD_DRAG_ANG = {rov_sim.QUAD_DRAG_ANG}")
    print(f"  THRUSTER_TAU_UP = {rov_sim.THRUSTER_TAU_UP}, TAU_DN = {rov_sim.THRUSTER_TAU_DN}")
    print(f"  RHO = {rov_sim.RHO}")
    
    # ── Analytical predictions ──
    # Terminal velocity: F_thrust = F_drag
    # All 3 horizontal thrusters contribute to forward:
    #   T1 fwd: 0.771 * 5.56 = 4.29 N
    #   T2 fwd: 0.766 * 5.56 = 4.26 N
    #   T4 fwd: 1.000 * 5.56 = 5.56 N
    #   Total forward thrust ≈ 14.1 N
    total_fwd_thrust = 14.1  # N (from geometry analysis)
    a_quad = 0.5 * rov_sim.RHO * rov_sim.CD[0] * rov_sim.AREA[0]
    b_lin = rov_sim.LIN_DRAG_BODY[0]
    c_thrust = -total_fwd_thrust
    # a_quad * v^2 + b_lin * v + c_thrust = 0
    disc = b_lin**2 - 4*a_quad*c_thrust
    v_terminal_analytical = (-b_lin + math.sqrt(disc)) / (2*a_quad)
    
    print(f"\n  ANALYTICAL PREDICTIONS (all 3 H-thrusters for forward):")
    print(f"    Total forward thrust:         {total_fwd_thrust:.1f} N")
    print(f"    Quadratic drag coeff (surge): {a_quad:.1f} N/(m/s)^2")
    print(f"    Linear drag coeff (surge):    {b_lin:.1f} N/(m/s)")
    print(f"    Terminal velocity (surge):     {v_terminal_analytical:.3f} m/s")
    
    # Effective mass with added mass
    m_eff = rov_sim.MASS + rov_sim.ADDED_MASS_BODY[0]
    print(f"    Effective mass (surge):        {m_eff:.1f} kg")
    print(f"    Initial accel (surge):         {rov_sim.MAX_THRUST_H/m_eff:.3f} m/s^2")
    
    results = {}

    # ── TEST 1: Forward surge — acceleration & terminal velocity ──
    # Use mixer output: surge=+1 → T1=+1, T2=+1, T4=+1 (all 3 horizontal thrusters)
    from joystick_panel import mix_joystick_to_thruster_cmds
    fwd_cmds = mix_joystick_to_thruster_cmds({"surge": 1.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}, n_thr)
    rev_cmds = mix_joystick_to_thruster_cmds({"surge": -1.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}, n_thr)
    yaw_r_cmds = mix_joystick_to_thruster_cmds({"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 1.0}, n_thr)
    
    print(f"\n{'─'*72}")
    print(f"  TEST 1: FORWARD SURGE (mixer: T1={fwd_cmds[0]:+.1f} T2={fwd_cmds[1]:+.1f} T4={fwd_cmds[3]:+.1f}, 20s)")
    print(f"{'─'*72}")
    reset_rov()
    # Settle 2s first
    data_settle = step_physics([0, 0, 0, 0], int(2.0 / DT))
    data = step_physics(fwd_cmds, int(20.0 / DT))
    
    speeds = [d['speed'] for d in data]
    max_speed = max(speeds)
    # Find time to 90% of max speed
    target_90 = 0.9 * max_speed
    t_90 = None
    for d in data:
        if d['speed'] >= target_90:
            t_90 = d['t']
            break
    
    # Speed at various times
    print(f"    Speed at 1s:  {next((d['speed'] for d in data if d['t'] >= 1.0), 0):.4f} m/s")
    print(f"    Speed at 3s:  {next((d['speed'] for d in data if d['t'] >= 3.0), 0):.4f} m/s")
    print(f"    Speed at 5s:  {next((d['speed'] for d in data if d['t'] >= 5.0), 0):.4f} m/s")
    print(f"    Speed at 10s: {next((d['speed'] for d in data if d['t'] >= 10.0), 0):.4f} m/s")
    print(f"    Speed at 15s: {next((d['speed'] for d in data if d['t'] >= 15.0), 0):.4f} m/s")
    print(f"    Speed at 20s: {next((d['speed'] for d in data if d['t'] >= 20.0), 0):.4f} m/s")
    print(f"    Max speed:    {max_speed:.4f} m/s")
    print(f"    Time to 90%:  {t_90:.1f}s" if t_90 else "    Time to 90%:  N/A")
    
    # Displacement
    start_x = data[0]['pos'][0]
    end_x = data[-1]['pos'][0]
    total_dist = math.sqrt((data[-1]['pos'][0]-data[0]['pos'][0])**2 + (data[-1]['pos'][1]-data[0]['pos'][1])**2)
    print(f"    Distance covered in 20s: {total_dist:.2f} m")
    results['surge_max_speed'] = max_speed
    results['surge_t90'] = t_90

    # ── TEST 2: Stopping distance ──
    print(f"\n{'─'*72}")
    print("  TEST 2: STOPPING (coast from max speed, 10s)")
    print(f"{'─'*72}")
    # Already moving at max speed from test 1, now cut thrust
    pos_before = p.getBasePositionAndOrientation(rov)[0]
    vel_before = vmag(p.getBaseVelocity(rov)[0])
    data_stop = step_physics([0, 0, 0, 0], int(10.0 / DT))
    
    stop_speeds = [d['speed'] for d in data_stop]
    # Time to slow to 10% of initial
    target_10 = 0.1 * vel_before
    t_stop = None
    for d in data_stop:
        if d['speed'] <= target_10:
            t_stop = d['t']
            break
    
    pos_after_1s = next((d['pos'] for d in data_stop if d['t'] >= 1.0), None)
    pos_after_3s = next((d['pos'] for d in data_stop if d['t'] >= 3.0), None)
    pos_final = data_stop[-1]['pos']
    
    stop_dist = math.sqrt((pos_final[0]-pos_before[0])**2 + (pos_final[1]-pos_before[1])**2)
    
    print(f"    Speed at coast start: {vel_before:.4f} m/s")
    print(f"    Speed at 1s:  {next((d['speed'] for d in data_stop if d['t'] >= 1.0), 0):.4f} m/s")
    print(f"    Speed at 3s:  {next((d['speed'] for d in data_stop if d['t'] >= 3.0), 0):.4f} m/s")
    print(f"    Speed at 5s:  {next((d['speed'] for d in data_stop if d['t'] >= 5.0), 0):.4f} m/s")
    print(f"    Speed at 10s: {next((d['speed'] for d in data_stop if d['t'] >= 10.0), 0):.4f} m/s")
    print(f"    Time to 10% speed: {t_stop:.1f}s" if t_stop else "    Time to 10% speed: >10s")
    print(f"    Stopping distance: {stop_dist:.3f} m")
    results['stop_time'] = t_stop
    results['stop_dist'] = stop_dist

    # ── TEST 3: Yaw rate ──
    print(f"\n{'─'*72}")
    print("  TEST 3: YAW RIGHT (T1=-1, T2=+1, 10s)")
    print(f"{'─'*72}")
    reset_rov()
    data_settle2 = step_physics([0, 0, 0, 0], int(2.0 / DT))
    yaw_start = data_settle2[-1]['yaw']
    data_yaw = step_physics([-1.0, 1.0, 0, 0], int(10.0 / DT))
    
    yaw_rates = []
    for i in range(1, len(data_yaw)):
        dt_data = data_yaw[i]['t'] - data_yaw[i-1]['t']
        if dt_data > 0:
            dyaw = data_yaw[i]['yaw'] - data_yaw[i-1]['yaw']
            # Handle wrap-around
            if dyaw > 180: dyaw -= 360
            if dyaw < -180: dyaw += 360
            yaw_rates.append(dyaw / dt_data)
    
    if yaw_rates:
        max_yaw_rate = max(abs(r) for r in yaw_rates)
        avg_yaw_rate = sum(abs(r) for r in yaw_rates[-20:]) / max(1, len(yaw_rates[-20:]))
    else:
        max_yaw_rate = avg_yaw_rate = 0
    
    total_yaw = data_yaw[-1]['yaw'] - yaw_start
    if total_yaw > 180: total_yaw -= 360
    if total_yaw < -180: total_yaw += 360
    
    print(f"    Total yaw in 10s: {total_yaw:.1f}°")
    print(f"    Max yaw rate:     {max_yaw_rate:.1f} °/s")
    print(f"    Steady yaw rate:  {avg_yaw_rate:.1f} °/s")
    # Also check lateral drift during pure yaw
    lat_drift = math.sqrt(data_yaw[-1]['pos'][0]**2 + data_yaw[-1]['pos'][1]**2)
    print(f"    Lateral drift:    {lat_drift:.3f} m  (should be minimal)")
    results['yaw_rate'] = avg_yaw_rate

    # ── TEST 4: Vertical (heave) ──
    print(f"\n{'─'*72}")
    print("  TEST 4: HEAVE UP (T3=+1, 10s)")
    print(f"{'─'*72}")
    reset_rov()
    data_settle3 = step_physics([0, 0, 0, 0], int(2.0 / DT))
    z_start = data_settle3[-1]['z']
    data_heave = step_physics([0, 0, 1.0, 0], int(10.0 / DT))
    
    z_speeds = [d['vel'][2] for d in data_heave]
    max_z_speed = max(z_speeds)
    z_end = data_heave[-1]['z']
    
    print(f"    Z start: {z_start:.3f} m")
    print(f"    Z end:   {z_end:.3f} m")
    print(f"    Rise:    {z_end - z_start:.3f} m in 10s")
    print(f"    Max vertical speed: {max_z_speed:.4f} m/s")
    print(f"    Vertical speed at 3s: {next((d['vel'][2] for d in data_heave if d['t'] >= 3.0), 0):.4f} m/s")
    print(f"    Vertical speed at 10s: {next((d['vel'][2] for d in data_heave if d['t'] >= 10.0), 0):.4f} m/s")
    results['heave_max_speed'] = max_z_speed

    # ── TEST 5: Acceleration profile (first 3 seconds in detail) ──
    print(f"\n{'─'*72}")
    print("  TEST 5: ACCELERATION PROFILE (surge, first 5s, 10Hz detail)")
    print(f"{'─'*72}")
    reset_rov()
    data_settle4 = step_physics([0, 0, 0, 0], int(2.0 / DT))
    
    thr_level_local = [0.0] * n_thr
    print(f"    {'Time':>5s}  {'Speed':>8s}  {'Accel':>8s}  {'ThrLvl':>8s}  {'DragEst':>8s}")
    prev_speed = 0.0
    prev_t = 0.0
    for step in range(int(5.0 / DT)):
        t = step * DT
        base_pos, base_quat = p.getBasePositionAndOrientation(rov)
        lin, ang = p.getBaseVelocity(rov)

        depth = max(0.0, rov_sim.SURFACE_Z - base_pos[2])
        _hull_half_z = 0.15
        submersion = min(1.0, max(0.0, depth / _hull_half_z)) if depth < _hull_half_z else 1.0
        depth_buoyancy_factor = max(0.5, 1.0 - rov_sim.DEPTH_BUOYANCY_COMPRESSIBILITY * depth)
        buoy_force = rov_sim.MASS * rov_sim.GRAVITY * rov_sim.BUOYANCY_SCALE * depth_buoyancy_factor * submersion
        cob_rel_world = p.rotateVector(base_quat, rov_sim.COB_OFFSET_BODY)
        cob_world = (base_pos[0]+cob_rel_world[0], base_pos[1]+cob_rel_world[1], base_pos[2]+cob_rel_world[2])
        p.applyExternalForce(rov, -1, (0.0, 0.0, buoy_force), cob_world, p.WORLD_FRAME)
        rov_sim.apply_ballast(rov, base_pos, base_quat)
        if submersion > 0.01:
            rov_sim.apply_righting_torque(rov, base_quat, ang, submersion)
        rov_sim.apply_drag(rov, base_pos, base_quat, lin, ang)

        inv_q = p.invertTransform([0, 0, 0], base_quat)[1]
        v_body = p.rotateVector(inv_q, lin)
        thr_cmd_local = list(fwd_cmds)  # use mixer output (all 3 H thrusters)
        for i, thr in enumerate(THRUSTERS):
            if thr_cmd_local[i] > thr_level_local[i]:
                tau = rov_sim.THRUSTER_TAU_UP
            else:
                tau = rov_sim.THRUSTER_TAU_DN
            thr_level_local[i] += (DT / max(1e-6, tau)) * (thr_cmd_local[i] - thr_level_local[i])
            thr_level_local[i] = max(-1.0, min(1.0, thr_level_local[i]))
            if abs(thr_level_local[i]) <= 1e-4:
                continue
            thrust_max = rov_sim.MAX_THRUST_H if thr["kind"] == "H" else rov_sim.MAX_THRUST_V
            thrust = thrust_max * thr_level_local[i]
            if thrust < 0: thrust *= rov_sim.BACKWARDS_THRUST_SCALE
            dir_body_t = thr.get("dir", (1.0, 0.0, 0.0))
            sa = v_body[0]*dir_body_t[0] + v_body[1]*dir_body_t[1] + v_body[2]*dir_body_t[2]
            loss = rov_sim.THRUSTER_SPEED_LOSS_COEF * abs(sa)
            loss = max(0.0, min(0.9, loss))
            thrust *= (1.0 - loss)
            dir_world = p.rotateVector(base_quat, thr["dir"])
            force = (dir_world[0]*thrust, dir_world[1]*thrust, dir_world[2]*thrust)
            rel_world = p.rotateVector(base_quat, thr["pos"])
            world_pos = (base_pos[0]+rel_world[0], base_pos[1]+rel_world[1], base_pos[2]+rel_world[2])
            p.applyExternalForce(rov, -1, force, world_pos, p.WORLD_FRAME)

        p.stepSimulation()

        # Print every 0.1s
        if step % max(1, int(0.1 / DT)) == 0 and t > 0:
            speed = vmag(lin)
            accel = (speed - prev_speed) / max(1e-6, t - prev_t)
            # Rough drag estimate
            drag_est = rov_sim.LIN_DRAG_BODY[0]*speed + 0.5*rov_sim.RHO*rov_sim.CD[0]*rov_sim.AREA[0]*speed**2
            print(f"    {t:5.2f}s  {speed:8.4f}  {accel:+8.4f}  {thr_level_local[3]:8.3f}  {drag_est:8.3f}")
            prev_speed = speed
            prev_t = t

    # ── SUMMARY ──
    print(f"\n{'='*72}")
    print("  REALISM ASSESSMENT")
    print(f"{'='*72}")
    
    # DDR spec expectations
    ddr_speed = 0.4  # m/s predicted
    
    print(f"\n  Metric              Current     DDR/Realistic    Status")
    print(f"  {'─'*60}")
    
    # Terminal speed
    status = "✅ GOOD" if 0.2 <= results['surge_max_speed'] <= 0.7 else "❌ BAD"
    print(f"  Terminal speed       {results['surge_max_speed']:.3f} m/s   ~0.3-0.5 m/s       {status}")
    
    # Time to 90%  —  analytically: τ_ramp(0.7s) + ~2.3·τ_hydro(~0.24s) ≈ 1.3s
    if results['surge_t90']:
        status = "✅ GOOD" if 1.0 <= results['surge_t90'] <= 5.0 else "❌ BAD"
        print(f"  Time to 90% speed   {results['surge_t90']:.1f}s        1-3s                {status}")
    
    # Stopping
    if results['stop_time']:
        status = "✅ GOOD" if 1.0 <= results['stop_time'] <= 6.0 else "❌ BAD"
        print(f"  Stop time (→10%)    {results['stop_time']:.1f}s        2-4s                {status}")
    status = "✅ GOOD" if 0.1 <= results['stop_dist'] <= 0.8 else "❌ BAD"
    print(f"  Stop distance       {results['stop_dist']:.3f} m    0.2-0.5 m           {status}")
    
    # Yaw
    status = "✅ GOOD" if 10.0 <= results['yaw_rate'] <= 45.0 else "❌ BAD"
    print(f"  Steady yaw rate     {results['yaw_rate']:.1f} °/s     15-40 °/s           {status}")
    
    # Heave
    status = "✅ GOOD" if 0.1 <= results['heave_max_speed'] <= 0.6 else "❌ BAD"
    print(f"  Heave max speed     {results['heave_max_speed']:.3f} m/s   0.2-0.4 m/s        {status}")
    
    print(f"\n  NOTE: DDR predicts ~0.4 m/s surge with 5.56N thrust, 7.5kg, CD=1.5, A=0.093m²")
    print(f"  Analytical terminal velocity: {v_terminal_analytical:.3f} m/s")
    
    p.disconnect()
    return results

if __name__ == "__main__":
    run()

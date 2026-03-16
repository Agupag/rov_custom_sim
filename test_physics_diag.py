#!/usr/bin/env python3
"""
Physics diagnostic: runs the REAL sim loop for 15 seconds in DIRECT mode
with full surge for 10s, then coast for 5s.  Logs detailed physics every
0.25s to a file AND stdout so we can see exactly what's happening.

This test catches problems the unit tests miss because it uses the actual
main-loop physics code path including constraints, indicators, environment, etc.
"""

import os, sys, math, time, json

LOG_PATH = os.path.join(os.path.dirname(__file__), "diag_physics.log")

def vmag(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def run():
    import pybullet as p
    import pybullet_data
    import joystick_panel as jp
    import rov_sim

    # ── Configure sim for headless run ──
    rov_sim.SLEEP_REALTIME = False
    rov_sim.HUD_ENABLED = False
    rov_sim.ENABLE_MARKERS = False
    rov_sim.ENABLE_CAMERA_PREVIEW = False
    rov_sim.LOG_PHYSICS_DETAILED = False

    # ── Joystick shared memory ──
    jp._ensure_shared()
    with jp._shared.get_lock():
        for i in range(6):
            jp._shared[i] = 0.0
        jp._shared[4] = 1.0  # active

    def set_js(surge=0.0, yaw=0.0):
        with jp._shared.get_lock():
            jp._shared[0] = surge
            jp._shared[3] = yaw
            jp._shared[4] = 1.0

    # ── PyBullet setup ──
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -rov_sim.GRAVITY)
    p.setPhysicsEngineParameter(fixedTimeStep=rov_sim.DT, numSolverIterations=50, numSubSteps=1)
    p.loadURDF("plane.urdf", [0, 0, rov_sim.SEABED_Z])

    # ── Build ROV ──
    rov, mesh_center = rov_sim.build_rov()
    obstacles = rov_sim.spawn_obstacles(rov_sim.NUM_OBSTACLES)

    # ── Detect thrusters ──
    auto_thr = rov_sim.detect_thrusters_from_gltf(rov_sim.GLTF_FILE, mesh_center)
    if auto_thr:
        rov_sim.THRUSTERS = auto_thr

    THRUSTERS = rov_sim.THRUSTERS
    n_thr = len(THRUSTERS)
    DT = rov_sim.DT

    # ── Create thruster indicators (same as main) ──
    thr_indicators = []
    if rov_sim.ENABLE_THRUSTER_ARROWS:
        thr_indicators = rov_sim.create_thruster_indicators(rov, THRUSTERS)

    # ── Thruster state ──
    thr_cmd = [0.0] * n_thr
    thr_level = [0.0] * n_thr
    thr_on = [False] * n_thr
    thr_reverse = [False] * n_thr
    js_last_sign = [0] * n_thr
    js_last_switch_t = [0.0] * n_thr

    rov_sim.LAST_VREL_BODY = None
    rov_sim.LAST_A_BODY = (0.0, 0.0, 0.0)

    # ── Phase schedule ──
    # 0-2s: settle, 2-12s: full surge, 12-15s: coast
    TOTAL_TIME = 15.0
    total_steps = int(round(TOTAL_TIME / DT))
    LOG_INTERVAL = 0.25  # seconds between log lines
    log_every = max(1, int(round(LOG_INTERVAL / DT)))

    # Open log
    logf = open(LOG_PATH, "w")
    header = ("step,time,phase,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,speed,"
              "omega_x,omega_y,omega_z,omega_mag,"
              "thr_cmd_0,thr_cmd_1,thr_cmd_2,thr_cmd_3,"
              "thr_lvl_0,thr_lvl_1,thr_lvl_2,thr_lvl_3,"
              "buoy_force,depth,submersion,"
              "drag_est_fx,drag_est_fy,drag_est_fz,"
              "num_contacts")
    logf.write(header + "\n")
    print(header)

    start_pos, _ = p.getBasePositionAndOrientation(rov)
    print(f"\n{'='*80}")
    print(f"  PHYSICS DIAGNOSTIC — 15s run, {total_steps} steps @ {1/DT:.0f}Hz")
    print(f"  ROV mass={rov_sim.MASS}kg, thrust_H={rov_sim.MAX_THRUST_H}N, thrust_V={rov_sim.MAX_THRUST_V}N")
    print(f"  Indicators: {len(thr_indicators)} constraint-attached bodies")
    print(f"  Thrusters: {n_thr}")
    for t in THRUSTERS:
        print(f"    {t['name']}: pos={t['pos']}, dir=({t['dir'][0]:+.3f},{t['dir'][1]:+.3f},{t['dir'][2]:+.3f}) kind={t['kind']}")
    print(f"  Start pos: {start_pos}")
    print(f"{'='*80}\n")

    for sim_step in range(total_steps):
        sim_t = sim_step * DT

        # Phase
        if sim_t < 2.0:
            phase = "settle"
            set_js(surge=0.0)
        elif sim_t < 12.0:
            phase = "surge"
            set_js(surge=1.0)
        else:
            phase = "coast"
            set_js(surge=0.0)

        # ── Read joystick → thruster commands ──
        js = jp.get_joystick_state()
        if js.get("active", False):
            js_cmds = jp.mix_joystick_to_thruster_cmds(js, n_thr)
            _now = time.monotonic()
            for i in range(n_thr):
                desired = js_cmds[i]
                if abs(desired) < 0.03:
                    d_sign = 0
                elif desired > 0:
                    d_sign = 1
                else:
                    d_sign = -1
                if d_sign != 0 and js_last_sign[i] != 0 and d_sign != js_last_sign[i]:
                    if (_now - js_last_switch_t[i]) < rov_sim.JOYSTICK_SWITCH_COOLDOWN:
                        desired = 0.0
                        d_sign = 0
                    else:
                        js_last_switch_t[i] = _now
                elif d_sign != js_last_sign[i]:
                    js_last_switch_t[i] = _now
                js_last_sign[i] = d_sign
                if abs(desired) < 0.03:
                    thr_cmd[i] = 0.0
                    thr_on[i] = False
                    thr_reverse[i] = False
                else:
                    thr_cmd[i] = desired
                    thr_on[i] = True
                    thr_reverse[i] = (desired < 0)

        # ── Physics ──
        base_pos, base_quat = p.getBasePositionAndOrientation(rov)
        lin, ang = p.getBaseVelocity(rov)

        depth = max(0.0, rov_sim.SURFACE_Z - base_pos[2])
        _hh = 0.15
        submersion = 1.0 if depth >= _hh else (0.0 if depth <= 0.0 else depth / _hh)
        dbf = max(0.5, 1.0 - rov_sim.DEPTH_BUOYANCY_COMPRESSIBILITY * depth)
        buoy = rov_sim.MASS * rov_sim.GRAVITY * rov_sim.BUOYANCY_SCALE * dbf * submersion
        cob_rel = p.rotateVector(base_quat, rov_sim.COB_OFFSET_BODY)
        cob_w = (base_pos[0]+cob_rel[0], base_pos[1]+cob_rel[1], base_pos[2]+cob_rel[2])
        p.applyExternalForce(rov, -1, (0, 0, buoy), cob_w, p.WORLD_FRAME)
        rov_sim.apply_ballast(rov, base_pos, base_quat)
        if submersion > 0.01:
            rov_sim.apply_righting_torque(rov, base_quat, ang, submersion)
        rov_sim.apply_drag(rov, base_pos, base_quat, lin, ang)

        rov_sim.apply_obstacle_water_forces(obstacles)

        # ── Thrusters ──
        inv_q = p.invertTransform([0,0,0], base_quat)[1]
        v_body = p.rotateVector(inv_q, lin)
        total_thrust_world = [0.0, 0.0, 0.0]
        for i, t in enumerate(THRUSTERS):
            tau = rov_sim.THRUSTER_TAU_UP if thr_cmd[i] > thr_level[i] else rov_sim.THRUSTER_TAU_DN
            thr_level[i] += (DT / max(1e-6, tau)) * (thr_cmd[i] - thr_level[i])
            thr_level[i] = rov_sim.clamp(thr_level[i], -1.0, 1.0)
            if abs(thr_level[i]) <= 1e-4:
                continue
            tmax = rov_sim.MAX_THRUST_H if t["kind"] == "H" else rov_sim.MAX_THRUST_V
            thrust = tmax * thr_level[i] * rov_sim.THRUST_LEVEL
            if thrust < 0:
                thrust *= rov_sim.BACKWARDS_THRUST_SCALE
            db = t.get("dir", (1,0,0))
            sa = v_body[0]*db[0]+v_body[1]*db[1]+v_body[2]*db[2]
            loss = rov_sim.clamp(rov_sim.THRUSTER_SPEED_LOSS_COEF * abs(sa), 0, 0.9)
            thrust *= (1.0 - loss)
            dw = p.rotateVector(base_quat, t["dir"])
            force = (dw[0]*thrust, dw[1]*thrust, dw[2]*thrust)
            rw = p.rotateVector(base_quat, t["pos"])
            wp = (base_pos[0]+rw[0], base_pos[1]+rw[1], base_pos[2]+rw[2])
            p.applyExternalForce(rov, -1, force, wp, p.WORLD_FRAME)
            total_thrust_world[0] += force[0]
            total_thrust_world[1] += force[1]
            total_thrust_world[2] += force[2]

        # Update indicators
        if thr_indicators:
            rov_sim.update_thruster_indicators(thr_indicators, base_pos, base_quat, thr_level)

        p.stepSimulation()

        # ── Count contacts (might explain stuck behavior) ──
        n_contacts = len(p.getContactPoints(bodyA=rov))

        # ── Log ──
        if sim_step % log_every == 0:
            speed = vmag(lin)
            omega_mag = vmag(ang)
            # Estimate drag force magnitude in body frame
            vrel = (lin[0] - rov_sim.WATER_CURRENT_WORLD[0],
                    lin[1] - rov_sim.WATER_CURRENT_WORLD[1],
                    lin[2] - rov_sim.WATER_CURRENT_WORLD[2])
            inv_q2 = p.invertTransform([0,0,0], base_quat)[1]
            vb = p.rotateVector(inv_q2, vrel)
            dfx = -rov_sim.LIN_DRAG_BODY[0]*vb[0] - 0.5*rov_sim.RHO*rov_sim.CD[0]*rov_sim.AREA[0]*abs(vb[0])*vb[0]
            dfy = -rov_sim.LIN_DRAG_BODY[1]*vb[1] - 0.5*rov_sim.RHO*rov_sim.CD[1]*rov_sim.AREA[1]*abs(vb[1])*vb[1]
            dfz = -rov_sim.LIN_DRAG_BODY[2]*vb[2] - 0.5*rov_sim.RHO*rov_sim.CD[2]*rov_sim.AREA[2]*abs(vb[2])*vb[2]

            line = (f"{sim_step},{sim_t:.3f},{phase},"
                    f"{base_pos[0]:.4f},{base_pos[1]:.4f},{base_pos[2]:.4f},"
                    f"{lin[0]:.4f},{lin[1]:.4f},{lin[2]:.4f},{speed:.4f},"
                    f"{ang[0]:.4f},{ang[1]:.4f},{ang[2]:.4f},{omega_mag:.4f},"
                    f"{thr_cmd[0]:.2f},{thr_cmd[1]:.2f},{thr_cmd[2]:.2f},{thr_cmd[3]:.2f},"
                    f"{thr_level[0]:.3f},{thr_level[1]:.3f},{thr_level[2]:.3f},{thr_level[3]:.3f},"
                    f"{buoy:.2f},{depth:.3f},{submersion:.3f},"
                    f"{dfx:.3f},{dfy:.3f},{dfz:.3f},"
                    f"{n_contacts}")
            logf.write(line + "\n")
            print(line)

    # ── Final summary ──
    end_pos, _ = p.getBasePositionAndOrientation(rov)
    end_lin, _ = p.getBaseVelocity(rov)
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    dz = end_pos[2] - start_pos[2]
    horiz = math.sqrt(dx*dx + dy*dy)
    
    print(f"\n{'='*80}")
    print(f"  FINAL RESULTS")
    print(f"{'='*80}")
    print(f"  Start: ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f})")
    print(f"  End:   ({end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f})")
    print(f"  Delta: dx={dx:.3f} dy={dy:.3f} dz={dz:.3f}")
    print(f"  Horizontal distance: {horiz:.3f}m")
    print(f"  Final speed: {vmag(end_lin):.4f} m/s")
    print(f"  Final thr_level: {[f'{x:.3f}' for x in thr_level]}")
    print(f"  Final thr_cmd:   {[f'{x:.2f}' for x in thr_cmd]}")
    
    # Verdict
    if horiz < 0.1:
        print(f"\n  ❌ FAIL: ROV barely moved ({horiz:.3f}m) — something is very wrong!")
        verdict = False
    elif horiz < 0.5:
        print(f"\n  ⚠️  WARN: ROV moved only {horiz:.3f}m in 10s of full thrust")
        verdict = False
    else:
        print(f"\n  ✅ PASS: ROV moved {horiz:.3f}m horizontally")
        verdict = True

    print(f"{'='*80}")
    logf.close()
    print(f"\nDetailed log: {LOG_PATH}")
    
    p.disconnect()

    # Clean up shared memory
    with jp._shared.get_lock():
        for i in range(6):
            jp._shared[i] = 0.0

    return verdict


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)

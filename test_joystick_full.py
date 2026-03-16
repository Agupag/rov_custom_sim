#!/usr/bin/env python3
"""
Full program autotest with simulated joystick movement.

Runs the real rov_sim physics (PyBullet DIRECT mode) while injecting
joystick axes through shared memory across multiple manoeuvre phases:

  Phase 0: Settle (2 s)          — no input, ROV finds buoyancy equilibrium
  Phase 1: Surge forward (3 s)   — left stick Y = +1.0
  Phase 2: Coast (1.5 s)         — release sticks
  Phase 3: Surge reverse (2 s)   — left stick Y = −1.0  (tests cooldown too)
  Phase 4: Yaw right (2 s)       — left stick X = +0.8
  Phase 5: Yaw left (2 s)        — left stick X = -0.7
  Phase 6: Heave up (2 s)        — heave button → T3 via joystick shared memory
  Phase 7: Combined (2 s)        — surge +0.6, yaw +0.4 simultaneously
  Phase 8: Cooldown stress (2 s) — rapid sign flips every 0.2 s (must be blocked)
  Phase 9: Settle (2 s)          — release, verify ROV decelerates

Left stick: surge (Y) → T1+T2+T4, yaw (X) → T1−T2 differential.  Binary ON/OFF mixer.
Heave buttons (▲/▼) → T3 vertical thruster via shared memory [2].

Run:  conda run -n rov_conda python test_joystick_full.py
"""

import os
import sys
import time
import math

# ── helpers ──────────────────────────────────────────────────────────
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def vmag(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def hmag(v):
    """Horizontal magnitude (XY only)."""
    return math.sqrt(v[0]**2 + v[1]**2)

# ── main test ────────────────────────────────────────────────────────
def run_test():
    import pybullet as p
    import pybullet_data
    import joystick_panel as jp

    # Set up shared memory (no subprocess — we drive it directly)
    jp._ensure_shared()
    with jp._shared.get_lock():
        for i in range(6):
            jp._shared[i] = 0.0
        jp._shared[4] = 1.0  # mark active

    def set_js(surge=0.0, sway=0.0, heave=0.0, yaw=0.0):
        with jp._shared.get_lock():
            jp._shared[0] = surge
            jp._shared[1] = sway
            jp._shared[2] = heave
            jp._shared[3] = yaw
            jp._shared[4] = 1.0

    def zero_js():
        set_js(0, 0, 0, 0)

    # ── Import sim modules ───────────────────────────────────────
    import rov_sim

    rov_sim.SLEEP_REALTIME = False
    rov_sim.HUD_ENABLED = False
    rov_sim.ENABLE_THRUSTER_ARROWS = False
    rov_sim.ENABLE_MARKERS = False
    rov_sim.ENABLE_CAMERA_PREVIEW = False
    rov_sim.LOG_PHYSICS_DETAILED = False

    # ── PyBullet setup (DIRECT mode — no window) ────────────────
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -rov_sim.GRAVITY)
    p.setPhysicsEngineParameter(fixedTimeStep=rov_sim.DT, numSolverIterations=50, numSubSteps=1)
    p.loadURDF("plane.urdf", [0, 0, rov_sim.SEABED_Z])

    rov, mesh_center = rov_sim.build_rov()
    obstacles = rov_sim.spawn_obstacles(rov_sim.NUM_OBSTACLES)

    # Detect thrusters (same as main())
    auto_thr = rov_sim.detect_thrusters_from_gltf(rov_sim.GLTF_FILE, mesh_center)
    if auto_thr:
        rov_sim.THRUSTERS = auto_thr

    THRUSTERS = rov_sim.THRUSTERS
    n_thr = len(THRUSTERS)
    DT = rov_sim.DT

    # Thruster state
    thr_cmd = [0.0] * n_thr
    thr_level = [0.0] * n_thr
    thr_on = [False] * n_thr
    thr_reverse = [False] * n_thr

    # Joystick cooldown state
    js_last_sign = [0] * n_thr
    js_last_switch_t = [0.0] * n_thr
    COOLDOWN = rov_sim.JOYSTICK_SWITCH_COOLDOWN

    # Added-mass state reset
    rov_sim.LAST_VREL_BODY = None
    rov_sim.LAST_A_BODY = (0.0, 0.0, 0.0)

    # ── Phase schedule ───────────────────────────────────────────
    phases = [
        # (name, duration_s, surge, sway, heave, yaw)
        # Left stick: surge (Y) → T1+T2+T4, yaw (X) → T1−T2 differential.  Binary ON/OFF mixer.
        # Heave/sway columns are kept for API compat but ignored by mixer.
        ("Settle",           2.0,  0.0,  0.0,  0.0,  0.0),
        ("Surge forward",    3.0,  1.0,  0.0,  0.0,  0.0),
        ("Coast",            1.5,  0.0,  0.0,  0.0,  0.0),
        ("Surge reverse",    2.0, -1.0,  0.0,  0.0,  0.0),
        ("Yaw right",        2.0,  0.0,  0.0,  0.0,  0.8),
        ("Yaw left",         2.0,  0.0,  0.0,  0.0, -0.7),
        ("Heave up",         2.0,  0.0,  0.0,  1.0,  0.0),  # heave via joystick buttons
        ("Combined",         2.0,  0.6,  0.0,  0.0,  0.4),
        ("Cooldown stress",  2.0,  0.0,  0.0,  0.0,  0.0),   # handled specially
        ("Final settle",     2.0,  0.0,  0.0,  0.0,  0.0),
    ]

    # Pre-compute step ranges
    phase_info = []  # (name, start_step, end_step, surge, sway, heave, yaw)
    step = 0
    for name, dur, su, sw, he, ya in phases:
        n_steps = int(round(dur / DT))
        phase_info.append((name, step, step + n_steps, su, sw, he, ya))
        step += n_steps
    total_steps = step

    # Storage for phase boundary snapshots
    snapshots = {}  # phase_name -> {"start": (pos, vel, omega), "end": (pos, vel, omega)}

    # ── Cooldown stress test tracking ────────────────────────────
    cooldown_flip_count = 0
    cooldown_blocked_count = 0

    # ── Main sim loop ────────────────────────────────────────────
    current_phase_idx = 0
    _cooldown_stress_sign = 1
    _cooldown_stress_last_flip = 0.0

    print("\n" + "=" * 70)
    print("  FULL PROGRAM AUTOTEST WITH JOYSTICK")
    print("=" * 70)
    print(f"  Thrusters: {n_thr} detected")
    for t in THRUSTERS:
        print(f"    {t['name']}: pos={t['pos']}, dir=({t['dir'][0]:+.3f},{t['dir'][1]:+.3f},{t['dir'][2]:+.3f}) kind={t['kind']}")
    print(f"  Total sim time: {total_steps * DT:.1f}s  ({total_steps} steps @ {1/DT:.0f}Hz)")
    print(f"  Cooldown: {COOLDOWN:.1f}s")
    print("=" * 70 + "\n")

    for sim_step in range(total_steps):
        sim_t = sim_step * DT

        # Determine current phase
        for pi, (pname, ps, pe, su, sw, he, ya) in enumerate(phase_info):
            if ps <= sim_step < pe:
                current_phase_idx = pi
                break

        pname, ps, pe, su, sw, he, ya = phase_info[current_phase_idx]

        # Record snapshot at phase start
        if sim_step == ps:
            pos_s, quat_s = p.getBasePositionAndOrientation(rov)
            lin_s, ang_s = p.getBaseVelocity(rov)
            snapshots.setdefault(pname, {})["start"] = (pos_s, lin_s, ang_s)
            print(f"  [{sim_t:6.2f}s] ▶ Phase: {pname:20s}  pos=({pos_s[0]:+6.2f},{pos_s[1]:+6.2f},{pos_s[2]:+6.2f})  |v|={vmag(lin_s):.3f}")

        # ── Set joystick axes ────────────────────────────────────
        if pname == "Cooldown stress":
            # Flip surge sign every 0.2 s (faster than 0.5 s cooldown)
            if sim_t - _cooldown_stress_last_flip >= 0.2:
                _cooldown_stress_sign *= -1
                _cooldown_stress_last_flip = sim_t
                cooldown_flip_count += 1
            set_js(surge=float(_cooldown_stress_sign), sway=0, heave=0, yaw=0)
        elif pname == "Heave up":
            set_js(surge=0, sway=0, heave=1.0, yaw=0)
        else:
            set_js(surge=su, sway=sw, heave=he, yaw=ya)

        # ── Read joystick → thruster commands ────────────────────
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
                    if (_now - js_last_switch_t[i]) < COOLDOWN:
                        desired = 0.0
                        d_sign = 0
                        if pname == "Cooldown stress":
                            cooldown_blocked_count += 1
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
                    thr_cmd[i] = desired   # -1 or +1 (binary ON/OFF)
                    thr_on[i] = True
                    thr_reverse[i] = (desired < 0)

        # ── Heave: read from joystick buttons (T3 = vertical) ────
        heave_cmd = js.get("heave", 0.0)
        if n_thr >= 3 and abs(heave_cmd) > 0.5:
            heave_sign = 1.0 if heave_cmd > 0 else -1.0
            thr_cmd[2] = heave_sign
            thr_on[2] = True
            thr_reverse[2] = (heave_sign < 0)
        elif n_thr >= 3 and abs(heave_cmd) <= 0.5:
            # Only zero T3 if mixer also left it zero
            if abs(thr_cmd[2]) < 0.01:
                thr_on[2] = False
                thr_reverse[2] = False

        # ── Physics ──────────────────────────────────────────────
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

        # Ocean current
        st = sim_step * DT
        rov_sim.WATER_CURRENT_WORLD[0] = rov_sim.WATER_CURRENT_BASE[0] + rov_sim.CURRENT_VARIATION_AMP[0] * math.sin(2*math.pi*st/rov_sim.CURRENT_VARIATION_PERIOD[0])
        rov_sim.WATER_CURRENT_WORLD[1] = rov_sim.WATER_CURRENT_BASE[1] + rov_sim.CURRENT_VARIATION_AMP[1] * math.sin(2*math.pi*st/rov_sim.CURRENT_VARIATION_PERIOD[1])
        rov_sim.WATER_CURRENT_WORLD[2] = rov_sim.WATER_CURRENT_BASE[2] + rov_sim.CURRENT_VARIATION_AMP[2] * math.cos(2*math.pi*st/rov_sim.CURRENT_VARIATION_PERIOD[2])

        rov_sim.apply_obstacle_water_forces(obstacles)

        # Thrusters
        inv_q = p.invertTransform([0,0,0], base_quat)[1]
        v_body = p.rotateVector(inv_q, lin)
        for i, t in enumerate(THRUSTERS):
            tau = rov_sim.THRUSTER_TAU_UP if thr_cmd[i] > thr_level[i] else rov_sim.THRUSTER_TAU_DN
            thr_level[i] += (DT / max(1e-6, tau)) * (thr_cmd[i] - thr_level[i])
            thr_level[i] = clamp(thr_level[i], -1.0, 1.0)
            if abs(thr_level[i]) <= 1e-4:
                continue
            tmax = rov_sim.MAX_THRUST_H if t["kind"] == "H" else rov_sim.MAX_THRUST_V
            thrust = tmax * thr_level[i] * rov_sim.THRUST_LEVEL
            if thrust < 0:
                thrust *= rov_sim.BACKWARDS_THRUST_SCALE
            db = t.get("dir", (1,0,0))
            sa = v_body[0]*db[0]+v_body[1]*db[1]+v_body[2]*db[2]
            loss = clamp(rov_sim.THRUSTER_SPEED_LOSS_COEF * abs(sa), 0, 0.9)
            thrust *= (1.0 - loss)
            dw = p.rotateVector(base_quat, t["dir"])
            force = (dw[0]*thrust, dw[1]*thrust, dw[2]*thrust)
            rw = p.rotateVector(base_quat, t["pos"])
            wp = (base_pos[0]+rw[0], base_pos[1]+rw[1], base_pos[2]+rw[2])
            p.applyExternalForce(rov, -1, force, wp, p.WORLD_FRAME)

        p.stepSimulation()

        # Record snapshot at phase end
        if sim_step == pe - 1:
            pos_e, quat_e = p.getBasePositionAndOrientation(rov)
            lin_e, ang_e = p.getBaseVelocity(rov)
            snapshots.setdefault(pname, {})["end"] = (pos_e, lin_e, ang_e)

    p.disconnect()

    # Zero out shared memory
    with jp._shared.get_lock():
        for i in range(6):
            jp._shared[i] = 0.0

    # ── Analyse results ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TEST RESULTS")
    print("=" * 70 + "\n")

    passed = 0
    failed = 0

    def check(ok, msg):
        nonlocal passed, failed
        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status}: {msg}")

    # --- Phase 0: Settle ---
    s = snapshots["Settle"]
    dz = abs(s["end"][0][2] - s["start"][0][2])
    speed_end = vmag(s["end"][1])
    check(dz < 0.3, f"Settle: Z drift {dz:.3f}m < 0.3m (buoyancy equilibrium)")
    check(speed_end < 0.5, f"Settle: final speed {speed_end:.3f} m/s < 0.5 (calm)")

    # --- Phase 1: Surge forward ---
    s = snapshots["Surge forward"]
    dp = (s["end"][0][0] - s["start"][0][0], s["end"][0][1] - s["start"][0][1])
    hdist = math.sqrt(dp[0]**2 + dp[1]**2)
    speed_end = vmag(s["end"][1])
    check(hdist > 0.05, f"Surge fwd: moved {hdist:.3f}m horizontally (>0.05m)")
    check(speed_end > 0.02, f"Surge fwd: final speed {speed_end:.3f} m/s (>0.02, still moving)")

    # --- Phase 2: Coast ---
    s = snapshots["Coast"]
    speed_start = vmag(s["start"][1])
    speed_end = vmag(s["end"][1])
    check(speed_end < speed_start, f"Coast: decelerated {speed_start:.3f} → {speed_end:.3f} m/s (drag)")

    # --- Phase 3: Surge reverse ---
    s = snapshots["Surge reverse"]
    # ROV should have slowed down and/or moved backwards
    vx_start = s["start"][1][0]
    vx_end = s["end"][1][0]
    # The velocity X component should be more negative (or less positive) at end
    # compared to start, because we applied reverse thrust
    hdist_rev = math.sqrt((s["end"][0][0]-s["start"][0][0])**2 + (s["end"][0][1]-s["start"][0][1])**2)
    check(hdist_rev > 0.01, f"Surge rev: moved {hdist_rev:.3f}m (>0.01m, thrust active)")

    # --- Phase 4: Yaw right ---
    s = snapshots["Yaw right"]
    # Expect angular velocity around Z (yaw rate) to have built up
    omega_z_end = s["end"][2][2]
    omega_mag = vmag(s["end"][2])
    check(omega_mag > 0.01, f"Yaw right: angular rate {omega_mag:.3f} rad/s (>0.01)")

    # --- Phase 5: Yaw left ---
    s = snapshots["Yaw left"]
    omega_mag_left = vmag(s["end"][2])
    check(omega_mag_left > 0.01, f"Yaw left: angular rate {omega_mag_left:.3f} rad/s (>0.01)")

    # --- Phase 6: Heave up (heave button → T3 via joystick) ---
    s = snapshots["Heave up"]
    dz_heave = s["end"][0][2] - s["start"][0][2]
    check(dz_heave > 0.01, f"Heave up: Z rose {dz_heave:.3f}m (>0.01m)")

    # --- Phase 7: Combined (surge + yaw) ---
    s = snapshots["Combined"]
    hdist_comb = math.sqrt((s["end"][0][0]-s["start"][0][0])**2 + (s["end"][0][1]-s["start"][0][1])**2)
    omega_comb = vmag(s["end"][2])
    check(hdist_comb > 0.01, f"Combined: horizontal movement {hdist_comb:.3f}m (>0.01m)")
    check(omega_comb > 0.005, f"Combined: angular rate {omega_comb:.3f} rad/s (>0.005, yaw active)")

    # --- Phase 8: Cooldown stress ---
    check(cooldown_flip_count >= 8,
          f"Cooldown stress: {cooldown_flip_count} sign flips attempted (>=8 in 2s @ 0.2s interval)")
    # With 0.5s cooldown and 0.2s flip interval, most flips should be blocked
    # At most 4 could succeed (at t=0, 0.5, 1.0, 1.5), rest blocked
    check(cooldown_blocked_count > 0,
          f"Cooldown stress: {cooldown_blocked_count} reversals blocked by cooldown (>0)")

    # --- Phase 9: Final settle ---
    s = snapshots["Final settle"]
    speed_start = vmag(s["start"][1])
    speed_end = vmag(s["end"][1])
    check(speed_end < speed_start + 0.1,
          f"Final settle: speed {speed_start:.3f} → {speed_end:.3f} m/s (not accelerating)")

    # --- Global checks ---
    # ROV should not have exploded (position within reasonable bounds)
    final_pos = snapshots["Final settle"]["end"][0]
    dist_from_origin = vmag(final_pos)
    check(dist_from_origin < 30.0,
          f"Stability: final position {dist_from_origin:.1f}m from origin (<30m, not exploded)")

    final_z = final_pos[2]
    check(final_z > rov_sim.SEABED_Z and final_z < rov_sim.SURFACE_Z + 1.0,
          f"Stability: final Z={final_z:.2f}m (between seabed {rov_sim.SEABED_Z} and surface {rov_sim.SURFACE_Z})")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    total = passed + failed
    print(f"  TOTAL: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  🎉 ALL TESTS PASSED")
    else:
        print("  ❌ SOME TESTS FAILED")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    ok = run_test()
    sys.exit(0 if ok else 1)

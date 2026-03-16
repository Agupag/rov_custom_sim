#!/usr/bin/env python3
"""
Joystick integration test for rov_sim.py.

This script:
  1. Starts the simulator in a subprocess (headless-ish via AUTOTEST mode
     but with joystick panel shared memory injected).
  2. Programmatically writes joystick axes to shared memory.
  3. Verifies that the thruster mixer produces the correct per-thruster
     commands.
  4. Verifies the 0.5 s direction-switch cooldown works.
  5. Verifies the ROV actually moves in the expected direction when
     joystick axes are applied.

Run:  conda run -n rov_conda python test_joystick.py
"""

import sys
import time
import math
import os

# ---------- Unit tests (no sim needed) ----------

def test_mixer():
    """Verify mix_joystick_to_thruster_cmds maps axes to thrusters correctly."""
    import joystick_panel as jp

    print("=" * 60)
    print("TEST 1: Thruster mixer mapping")
    print("=" * 60)

    cases = [
        # (surge, sway, heave, yaw) -> expected (T1, T2, T3, T4)
        #
        # Binary ON/OFF mixer: raw_T1 = surge - yaw, raw_T2 = surge + yaw, raw_T4 = surge
        # Each raw value is snapped: |val|<0.15 → 0, val>0 → +1, val<0 → -1
        # T1 cmd=+1 → yaw torque CCW (left), so yaw_right means decrease T1
        # Dead zone: overall magnitude < 0.15 → all OFF.
        # T3 (heave) is keyboard-only → always 0 from mixer.

        ({"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0},
         [0.0, 0.0, 0.0, 0.0], "Dead zone — all zero"),

        ({"surge": 1.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0},
         [1.0, 1.0, 0.0, 1.0], "Full forward — all H thrusters ON"),

        ({"surge": -1.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0},
         [-1.0, -1.0, 0.0, -1.0], "Full reverse — all H thrusters REV"),

        ({"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 1.0},
         [-1.0, 1.0, 0.0, 0.0], "Full yaw right — T1=−1 T2=+1"),

        ({"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": -1.0},
         [1.0, -1.0, 0.0, 0.0], "Full yaw left — T1=+1 T2=−1"),

        # Heave and sway are keyboard-only, mixer ignores them
        ({"surge": 0.0, "sway": 0.0, "heave": 1.0, "yaw": 0.0},
         [0.0, 0.0, 0.0, 0.0], "Heave ignored — T3=0 (keyboard-only)"),

        ({"surge": 0.0, "sway": 1.0, "heave": 0.0, "yaw": 0.0},
         [0.0, 0.0, 0.0, 0.0], "Sway ignored — all 0 (keyboard-only)"),

        # Binary snap: surge 0.8 + yaw 0.5
        # raw_T1 = 0.8-0.5 = 0.3 → +1, raw_T2 = 0.8+0.5 = 1.3 → +1, raw_T4 = 0.8 → +1
        ({"surge": 0.8, "sway": 0.0, "heave": 0.0, "yaw": 0.5},
         [1.0, 1.0, 0.0, 1.0], "Fwd + yaw right — all ON (raw T1=0.3 snaps to +1)"),

        # Below dead zone
        ({"surge": 0.1, "sway": 0.0, "heave": 0.0, "yaw": 0.0},
         [0.0, 0.0, 0.0, 0.0], "Surge 0.1 — below dead zone → all OFF"),

        # Forward + yaw left: raw_T1 = 0.5+0.5 = 1.0 → +1, raw_T2 = 0.5-0.5 = 0.0 → 0, raw_T4 = 0.5 → +1
        ({"surge": 0.5, "sway": -0.3, "heave": 0.8, "yaw": -0.5},
         [1.0, 0.0, 0.0, 1.0], "Fwd + yaw left — T1=+1 T2=OFF T4=+1"),

        # Reverse + yaw right: raw_T1 = -0.7-0.7 = -1.4 → -1, raw_T2 = -0.7+0.7 = 0.0 → 0, raw_T4 = -0.7 → -1
        ({"surge": -0.7, "sway": 0.0, "heave": 0.0, "yaw": 0.7},
         [-1.0, 0.0, 0.0, -1.0], "Rev + yaw right — T1=-1 T2=OFF T4=-1"),

        # Reverse + yaw left: raw_T1 = -0.7+0.7 = 0.0 → 0, raw_T2 = -0.7-0.7 = -1.4 → -1, raw_T4 = -0.7 → -1
        ({"surge": -0.7, "sway": 0.0, "heave": 0.0, "yaw": -0.7},
         [0.0, -1.0, 0.0, -1.0], "Rev + yaw left — T1=OFF T2=-1 T4=-1"),
    ]

    passed = 0
    failed = 0
    for state, expected, label in cases:
        cmds = jp.mix_joystick_to_thruster_cmds(state, 4)
        ok = all(abs(cmds[i] - expected[i]) < 0.01 for i in range(4))
        status = "✅ PASS" if ok else "❌ FAIL"
        if not ok:
            failed += 1
        else:
            passed += 1
        print(f"  {status}: {label}")
        if not ok:
            print(f"         Expected: {expected}")
            print(f"         Got:      {cmds}")

    print(f"\n  Mixer tests: {passed} passed, {failed} failed\n")
    return failed == 0


def test_shared_memory():
    """Verify shared memory read/write works correctly."""
    import joystick_panel as jp

    print("=" * 60)
    print("TEST 2: Shared memory IPC")
    print("=" * 60)

    jp._ensure_shared()

    # Write values
    with jp._shared.get_lock():
        jp._shared[0] = 0.75   # surge
        jp._shared[1] = -0.25  # sway
        jp._shared[2] = 0.5    # heave
        jp._shared[3] = -0.1   # yaw
        jp._shared[4] = 1.0    # active

    state = jp.get_joystick_state()

    checks = [
        (abs(state["surge"] - 0.75) < 0.001, f"surge={state['surge']}, expected 0.75"),
        (abs(state["sway"] - (-0.25)) < 0.001, f"sway={state['sway']}, expected -0.25"),
        (abs(state["heave"] - 0.5) < 0.001, f"heave={state['heave']}, expected 0.5"),
        (abs(state["yaw"] - (-0.1)) < 0.001, f"yaw={state['yaw']}, expected -0.1"),
        (state["active"] is True, f"active={state['active']}, expected True"),
    ]

    passed = 0
    failed = 0
    for ok, msg in checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        if not ok:
            failed += 1
        else:
            passed += 1
        print(f"  {status}: {msg}")

    # Reset
    with jp._shared.get_lock():
        for i in range(6):
            jp._shared[i] = 0.0

    state2 = jp.get_joystick_state()
    ok = state2["active"] is False and abs(state2["surge"]) < 0.001
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  {status}: reset to zero — active={state2['active']}, surge={state2['surge']}")

    print(f"\n  Shared memory tests: {passed} passed, {failed} failed\n")
    return failed == 0


def test_cooldown_logic():
    """Verify the 0.5 s direction-switch cooldown behaves correctly."""
    print("=" * 60)
    print("TEST 3: Direction-switch cooldown logic")
    print("=" * 60)

    COOLDOWN = 0.5
    # Simulate the cooldown state machine from rov_sim.py
    last_sign = [0] * 4
    last_switch_t = [0.0] * 4

    def apply_cmd(i, desired, now):
        """Replicate the cooldown logic from rov_sim main loop."""
        if abs(desired) < 0.03:
            desired_sign = 0
        elif desired > 0:
            desired_sign = 1
        else:
            desired_sign = -1

        if desired_sign != 0 and last_sign[i] != 0 and desired_sign != last_sign[i]:
            if (now - last_switch_t[i]) < COOLDOWN:
                desired = 0.0
                desired_sign = 0
            else:
                last_switch_t[i] = now
        elif desired_sign != last_sign[i]:
            last_switch_t[i] = now

        last_sign[i] = desired_sign
        return desired

    passed = 0
    failed = 0

    # Test: forward command at t=0
    result = apply_cmd(0, 1.0, 0.0)
    ok = abs(result - 1.0) < 0.01
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok: passed += 1
    else: failed += 1
    print(f"  {status}: t=0.0s  cmd=+1.0  → {result:.2f} (expect +1.0, first command)")

    # Test: reverse at t=0.2 (within cooldown) — should be blocked
    result = apply_cmd(0, -1.0, 0.2)
    ok = abs(result) < 0.01
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok: passed += 1
    else: failed += 1
    print(f"  {status}: t=0.2s  cmd=-1.0  → {result:.2f} (expect 0.0, cooldown blocks)")

    # Test: reverse at t=0.6 (after cooldown) — should be allowed
    result = apply_cmd(0, -1.0, 0.6)
    ok = abs(result - (-1.0)) < 0.01
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok: passed += 1
    else: failed += 1
    print(f"  {status}: t=0.6s  cmd=-1.0  → {result:.2f} (expect -1.0, cooldown expired)")

    # Test: forward again at t=0.8 (within cooldown of last switch) — should be blocked
    result = apply_cmd(0, 1.0, 0.8)
    ok = abs(result) < 0.01
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok: passed += 1
    else: failed += 1
    print(f"  {status}: t=0.8s  cmd=+1.0  → {result:.2f} (expect 0.0, cooldown blocks)")

    # Test: forward at t=1.2 (cooldown from t=0.6 expired) — should be allowed
    result = apply_cmd(0, 1.0, 1.2)
    ok = abs(result - 1.0) < 0.01
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok: passed += 1
    else: failed += 1
    print(f"  {status}: t=1.2s  cmd=+1.0  → {result:.2f} (expect +1.0, cooldown expired)")

    # Test: zero command always passes regardless of cooldown
    result = apply_cmd(0, 0.0, 1.2)
    ok = abs(result) < 0.01
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok: passed += 1
    else: failed += 1
    print(f"  {status}: t=1.2s  cmd= 0.0  → {result:.2f} (expect 0.0, zero always OK)")

    print(f"\n  Cooldown tests: {passed} passed, {failed} failed\n")
    return failed == 0


def test_panel_subprocess():
    """Verify the panel subprocess starts, communicates, and stops."""
    import joystick_panel as jp

    print("=" * 60)
    print("TEST 4: Panel subprocess lifecycle")
    print("=" * 60)

    passed = 0
    failed = 0

    # Start panel
    jp.start_joystick_panel()
    time.sleep(1.0)

    state = jp.get_joystick_state()
    ok = state["active"] is True
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok: passed += 1
    else: failed += 1
    print(f"  {status}: Panel launched, active={state['active']}")

    # Write axes via shared memory (simulating user dragging the joystick)
    with jp._shared.get_lock():
        jp._shared[0] = 0.6  # surge
        jp._shared[3] = 0.4  # yaw
    time.sleep(0.15)

    state = jp.get_joystick_state()
    ok = abs(state["surge"] - 0.6) < 0.01 and abs(state["yaw"] - 0.4) < 0.01
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok: passed += 1
    else: failed += 1
    print(f"  {status}: Wrote surge=0.6 yaw=0.4, read back surge={state['surge']:.2f} yaw={state['yaw']:.2f}")

    # Verify mixer output for these axes (proportional mixer)
    cmds = jp.mix_joystick_to_thruster_cmds(state, 4)
    # surge=0.6 yaw=0.4: T1=0.6-0.4=0.2, T2=0.6+0.4=1.0, T4=0.6
    ok = abs(cmds[0] - 0.2) < 0.01 and abs(cmds[1] - 1.0) < 0.01 and abs(cmds[3] - 0.6) < 0.01
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok: passed += 1
    else: failed += 1
    print(f"  {status}: Mixer → T1={cmds[0]:.2f} T2={cmds[1]:.2f} T4={cmds[3]:.2f} (expect 0.2, 1.0, 0.6 — proportional blend)")

    # Stop panel
    jp.stop_joystick_panel()
    time.sleep(0.5)

    state = jp.get_joystick_state()
    ok = state["active"] is False
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok: passed += 1
    else: failed += 1
    print(f"  {status}: Panel stopped, active={state['active']}")

    # Verify process is dead
    ok = jp._process is None or not jp._process.is_alive()
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok: passed += 1
    else: failed += 1
    print(f"  {status}: Process terminated cleanly")

    print(f"\n  Subprocess tests: {passed} passed, {failed} failed\n")
    return failed == 0


def test_sim_integration():
    """
    Run the actual simulator for a few seconds with programmatic joystick
    input and verify the ROV moves in the expected direction.
    
    We inject joystick commands via a background thread that writes to
    shared memory while the sim runs.
    """
    import subprocess
    import threading
    import joystick_panel as jp

    print("=" * 60)
    print("TEST 5: Full sim integration (ROV movement from joystick)")
    print("=" * 60)

    # We'll run the sim as a subprocess with AUTOTEST=0 but a custom
    # short-lived script that:
    #   1. Imports rov_sim
    #   2. Starts the joystick shared memory
    #   3. Injects surge=+1.0 for 3 seconds
    #   4. Records final position
    #   5. Checks ROV moved forward (+X)

    test_script = '''
import os, sys, time, math
os.environ["ROV_AUTOTEST"] = "0"  # not autotest mode

import joystick_panel as jp

# Pre-create shared memory so rov_sim picks it up
jp._ensure_shared()

# Set joystick active with full surge forward
with jp._shared.get_lock():
    jp._shared[0] = 1.0   # surge
    jp._shared[1] = 0.0   # sway
    jp._shared[2] = 0.0   # heave
    jp._shared[3] = 0.0   # yaw
    jp._shared[4] = 1.0   # active

import pybullet as p
import pybullet_data
import rov_sim

# Monkey-patch: skip the joystick panel subprocess launch since we
# already have shared memory set up, and skip the GUI
_orig_connect = p.connect
def _fake_connect(mode, *a, **kw):
    return _orig_connect(p.DIRECT, *a, **kw)
p.connect = _fake_connect

# Also need to suppress addUserDebugText/Line in DIRECT mode
_orig_addText = p.addUserDebugText
def _safe_addText(*a, **kw):
    try: return _orig_addText(*a, **kw)
    except: return -1
p.addUserDebugText = _safe_addText

_orig_addLine = p.addUserDebugLine
def _safe_addLine(*a, **kw):
    try: return _orig_addLine(*a, **kw)
    except: return -1
p.addUserDebugLine = _safe_addLine

_orig_configVis = p.configureDebugVisualizer
def _safe_configVis(*a, **kw):
    try: return _orig_configVis(*a, **kw)
    except: pass
p.configureDebugVisualizer = _safe_configVis

_orig_resetCam = p.resetDebugVisualizerCamera
def _safe_resetCam(*a, **kw):
    try: return _orig_resetCam(*a, **kw)
    except: pass
p.resetDebugVisualizerCamera = _safe_resetCam

_orig_getKeys = p.getKeyboardEvents
def _no_keys():
    return {}
p.getKeyboardEvents = _no_keys

# We'll run the main loop manually for N steps
rov_sim.SLEEP_REALTIME = False
rov_sim.HUD_ENABLED = False
rov_sim.ENABLE_THRUSTER_ARROWS = False
rov_sim.ENABLE_MARKERS = False
rov_sim.ENABLE_CAMERA_PREVIEW = False
rov_sim.LOG_PHYSICS_DETAILED = False

# Do a shortened version: connect, build, run N steps
_orig_p_connect = p.connect  # already patched
p.connect = _orig_connect  # restore for actual use
cid = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -rov_sim.GRAVITY)
p.setPhysicsEngineParameter(fixedTimeStep=rov_sim.DT, numSolverIterations=50, numSubSteps=1)
p.loadURDF("plane.urdf", [0, 0, rov_sim.SEABED_Z])

rov, mesh_center = rov_sim.build_rov()
obstacles = rov_sim.spawn_obstacles(rov_sim.NUM_OBSTACLES)

# Detect thrusters
auto_thr = rov_sim.detect_thrusters_from_gltf(rov_sim.GLTF_FILE, mesh_center)
if auto_thr:
    rov_sim.THRUSTERS = auto_thr

THRUSTERS = rov_sim.THRUSTERS
n_thr = len(THRUSTERS)
thr_cmd = [0.0] * n_thr
thr_level = [0.0] * n_thr
thr_on = [False] * n_thr
thr_reverse = [False] * n_thr

_js_last_sign = [0] * n_thr
_js_last_switch_t = [0.0] * n_thr

# Record start position
start_pos, _ = p.getBasePositionAndOrientation(rov)
print(f"START_POS: {start_pos[0]:.4f} {start_pos[1]:.4f} {start_pos[2]:.4f}")

# Run 360 steps (3 seconds at 120Hz) with full surge
N_STEPS = 360
for step in range(N_STEPS):
    # Read joystick
    js = jp.get_joystick_state()
    if js.get("active", False):
        js_cmds = jp.mix_joystick_to_thruster_cmds(js, n_thr)
        _now = time.monotonic()
        for i in range(n_thr):
            desired = js_cmds[i]
            if abs(desired) < 0.03:
                desired_sign = 0
            elif desired > 0:
                desired_sign = 1
            else:
                desired_sign = -1
            if desired_sign != 0 and _js_last_sign[i] != 0 and desired_sign != _js_last_sign[i]:
                if (_now - _js_last_switch_t[i]) < 0.5:
                    desired = 0.0
                    desired_sign = 0
                else:
                    _js_last_switch_t[i] = _now
            elif desired_sign != _js_last_sign[i]:
                _js_last_switch_t[i] = _now
            _js_last_sign[i] = desired_sign
            if abs(desired) < 0.03:
                thr_cmd[i] = 0.0
                thr_on[i] = False
                thr_reverse[i] = False
            else:
                thr_cmd[i] = desired
                thr_on[i] = True
                thr_reverse[i] = (desired < 0)

    # Physics
    base_pos, base_quat = p.getBasePositionAndOrientation(rov)
    lin, ang = p.getBaseVelocity(rov)

    depth = max(0.0, rov_sim.SURFACE_Z - base_pos[2])
    _hull_half_z = 0.15
    if depth >= _hull_half_z: submersion = 1.0
    elif depth <= 0.0: submersion = 0.0
    else: submersion = depth / _hull_half_z

    depth_buoy_factor = max(0.5, 1.0 - rov_sim.DEPTH_BUOYANCY_COMPRESSIBILITY * depth)
    buoy_force = rov_sim.MASS * rov_sim.GRAVITY * rov_sim.BUOYANCY_SCALE * depth_buoy_factor * submersion
    cob_rel = p.rotateVector(base_quat, rov_sim.COB_OFFSET_BODY)
    cob_w = (base_pos[0]+cob_rel[0], base_pos[1]+cob_rel[1], base_pos[2]+cob_rel[2])
    p.applyExternalForce(rov, -1, (0,0,buoy_force), cob_w, p.WORLD_FRAME)
    rov_sim.apply_ballast(rov, base_pos, base_quat)
    if submersion > 0.01:
        rov_sim.apply_righting_torque(rov, base_quat, ang, submersion)
    rov_sim.apply_drag(rov, base_pos, base_quat, lin, ang)
    rov_sim.apply_obstacle_water_forces(obstacles)

    # Thrusters
    inv_q = p.invertTransform([0,0,0], base_quat)[1]
    v_body = p.rotateVector(inv_q, lin)
    for i, t in enumerate(THRUSTERS):
        if thr_cmd[i] > thr_level[i]: tau = rov_sim.THRUSTER_TAU_UP
        else: tau = rov_sim.THRUSTER_TAU_DN
        thr_level[i] += (rov_sim.DT / max(1e-6, tau)) * (thr_cmd[i] - thr_level[i])
        thr_level[i] = rov_sim.clamp(thr_level[i], -1.0, 1.0)
        if abs(thr_level[i]) <= 1e-4: continue
        thrust_max = rov_sim.MAX_THRUST_H if t["kind"]=="H" else rov_sim.MAX_THRUST_V
        thrust = thrust_max * thr_level[i] * rov_sim.THRUST_LEVEL
        if thrust < 0: thrust *= rov_sim.BACKWARDS_THRUST_SCALE
        dir_body = t.get("dir", (1,0,0))
        speed_along = v_body[0]*dir_body[0]+v_body[1]*dir_body[1]+v_body[2]*dir_body[2]
        loss = rov_sim.clamp(rov_sim.THRUSTER_SPEED_LOSS_COEF*abs(speed_along), 0, 0.9)
        thrust *= (1.0-loss)
        dir_w = p.rotateVector(base_quat, t["dir"])
        force = (dir_w[0]*thrust, dir_w[1]*thrust, dir_w[2]*thrust)
        rel_w = p.rotateVector(base_quat, t["pos"])
        wp = (base_pos[0]+rel_w[0], base_pos[1]+rel_w[1], base_pos[2]+rel_w[2])
        p.applyExternalForce(rov, -1, force, wp, p.WORLD_FRAME)

    p.stepSimulation()

end_pos, _ = p.getBasePositionAndOrientation(rov)
print(f"END_POS: {end_pos[0]:.4f} {end_pos[1]:.4f} {end_pos[2]:.4f}")
print(f"THR_LEVELS: {' '.join(f'{x:.3f}' for x in thr_level)}")
print(f"THR_CMDS: {' '.join(f'{x:.3f}' for x in thr_cmd)}")

dx = end_pos[0] - start_pos[0]
dy = end_pos[1] - start_pos[1]
dz = end_pos[2] - start_pos[2]
print(f"DELTA: dx={dx:.4f} dy={dy:.4f} dz={dz:.4f}")

p.disconnect()

# Clean up shared memory
with jp._shared.get_lock():
    for i in range(6):
        jp._shared[i] = 0.0
'''

    # Run as subprocess
    import subprocess
    result = subprocess.run(
        ["conda", "run", "-n", "rov_conda", "python", "-c", test_script],
        capture_output=True, text=True, timeout=30,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    stdout = result.stdout
    stderr = result.stderr
    
    if result.returncode != 0:
        print(f"  ❌ FAIL: Sim subprocess exited with code {result.returncode}")
        print(f"  STDOUT: {stdout[:500]}")
        print(f"  STDERR: {stderr[:500]}")
        return False

    # Parse results
    lines = stdout.strip().split("\n")
    results = {}
    for line in lines:
        for prefix in ("START_POS:", "END_POS:", "DELTA:", "THR_LEVELS:", "THR_CMDS:"):
            if line.strip().startswith(prefix):
                results[prefix.rstrip(":")] = line.strip()[len(prefix):].strip()

    passed = 0
    failed = 0

    # Check thruster commands were applied (surge=1 → T1=1 T2=1 T3=0 T4=1, proportional mixer)
    if "THR_CMDS" in results:
        cmds = [float(x) for x in results["THR_CMDS"].split()]
        ok = abs(cmds[0] - 1.0) < 0.01 and abs(cmds[1] - 1.0) < 0.01 and abs(cmds[2]) < 0.01 and abs(cmds[3] - 1.0) < 0.01
        status = "✅ PASS" if ok else "❌ FAIL"
        if ok: passed += 1
        else: failed += 1
        print(f"  {status}: Thruster commands T1={cmds[0]:.2f} T2={cmds[1]:.2f} T3={cmds[2]:.2f} T4={cmds[3]:.2f}")
        print(f"           Expected:          T1=1.00 T2=1.00 T3=0.00 T4=1.00")
    else:
        print(f"  ❌ FAIL: Could not parse THR_CMDS from output")
        failed += 1

    # Check thruster levels ramped up (T4 is the surge thruster now)
    if "THR_LEVELS" in results:
        levels = [float(x) for x in results["THR_LEVELS"].split()]
        ok = levels[3] > 0.9  # T4 should have ramped up from surge
        status = "✅ PASS" if ok else "❌ FAIL"
        if ok: passed += 1
        else: failed += 1
        print(f"  {status}: Thruster levels ramped: T4={levels[3]:.3f} (expect >0.9)")
    else:
        print(f"  ❌ FAIL: Could not parse THR_LEVELS from output")
        failed += 1

    # Check ROV moved forward (positive delta, direction depends on thruster orientations)
    if "DELTA" in results:
        parts = results["DELTA"].split()
        dx = float(parts[0].split("=")[1])
        dy = float(parts[1].split("=")[1])
        dz = float(parts[2].split("=")[1])
        horiz_dist = math.sqrt(dx*dx + dy*dy)
        ok = horiz_dist > 0.04  # Should have moved at least 0.04m in 3 seconds with DDR-spec thrust (5.56N)
        status = "✅ PASS" if ok else "❌ FAIL"
        if ok: passed += 1
        else: failed += 1
        print(f"  {status}: ROV moved horizontally {horiz_dist:.3f}m (expect >0.04m) — dx={dx:.3f} dy={dy:.3f} dz={dz:.3f}")

        # Z should be roughly stable (buoyancy holds it)
        ok = abs(dz) < 1.0
        status = "✅ PASS" if ok else "❌ FAIL"
        if ok: passed += 1
        else: failed += 1
        print(f"  {status}: Z drift = {dz:.3f}m (expect <1.0m — buoyancy holds depth)")
    else:
        print(f"  ❌ FAIL: Could not parse DELTA from output")
        failed += 2

    print(f"\n  Sim integration tests: {passed} passed, {failed} failed\n")
    return failed == 0


# ---------- Main ----------
if __name__ == "__main__":
    print("\n" + "🎮 " * 20)
    print("JOYSTICK INTEGRATION TEST SUITE")
    print("🎮 " * 20 + "\n")

    all_pass = True
    all_pass &= test_mixer()
    all_pass &= test_shared_memory()
    all_pass &= test_cooldown_logic()
    all_pass &= test_panel_subprocess()
    all_pass &= test_sim_integration()

    print("=" * 60)
    if all_pass:
        print("🎉  ALL TESTS PASSED")
    else:
        print("❌  SOME TESTS FAILED")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)

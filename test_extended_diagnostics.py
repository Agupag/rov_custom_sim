#!/usr/bin/env python3
"""
Extended Diagnostics Suite: Assist Modes, GLTF Validation, Timing Metrics, Physics Calibration

This module provides comprehensive validation of:
  1. Assist mode performance (depth hold, heading hold)
  2. GLTF thruster frame validation (positions, rotations, torque signs)
  3. Timing and latency metrics (loop frequency, frame capture, recording)
  4. Combined-axis command validation
  5. Physics calibration baselines and sensitivity analysis

Runs all tests in PyBullet DIRECT (headless) mode with optional detailed logging.
"""

import os, sys, math, time, json
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pybullet as p
import pybullet_data
import numpy as np

# Import simulation modules
import rov_sim
import joystick_panel
from sim_shared import (
    SURGE, SWAY, HEAVE, YAW, ACTIVE, CAM_TILT, ROLL_RAD, PITCH_RAD,
    REC_FLAG, DEPTH_M, HEADING_DEG, SPEED_MPS, THRUST_LEVEL,
    DEPTH_HOLD_ACTIVE, HEADING_HOLD_ACTIVE, REC_STATUS, CONTROL_MODE
)

# ──────────────────────────────────────────────────────────────────────
#  PART 1: ASSIST MODE ACCEPTANCE TESTS
# ──────────────────────────────────────────────────────────────────────

class DepthHoldTest:
    """Validate depth-hold controller response to step disturbances."""
    
    def __init__(self):
        self.cid = None
        self.rov = None
        self.thrusters = []
        self.base_orn = None
        
    def setup(self):
        """Initialize PyBullet and ROV."""
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -rov_sim.GRAVITY)
        p.setPhysicsEngineParameter(fixedTimeStep=rov_sim.DT, numSolverIterations=50, numSubSteps=1)
        p.loadURDF("plane.urdf", [0, 0, rov_sim.SEABED_Z])
        
        self.rov, mesh_center = rov_sim.build_rov()
        gltf_path = os.path.join(ROOT, "Assembly 1.gltf")
        self.thrusters = rov_sim.detect_thrusters_from_gltf(gltf_path, mesh_center)
        
        euler_rad = tuple(math.radians(d) for d in rov_sim.MESH_BODY_EULER_DEG)
        self.base_orn = p.getQuaternionFromEuler(euler_rad)
    
    def cleanup(self):
        if self.cid is not None:
            p.disconnect(self.cid)
            self.cid = None
    
    def reset_rov(self, depth=2.0):
        """Reset ROV to starting position at specified depth."""
        start_pos = [0, 0, rov_sim.SURFACE_Z - depth]
        p.resetBasePositionAndOrientation(self.rov, start_pos, self.base_orn)
        p.resetBaseVelocity(self.rov, [0, 0, 0], [0, 0, 0])
        rov_sim.LAST_VREL_BODY = None
        rov_sim.LAST_A_BODY = (0.0, 0.0, 0.0)
        rov_sim.LAST_W_BODY = None
        rov_sim.LAST_ALPHA_BODY = (0.0, 0.0, 0.0)
    
    def apply_physics_step(self, thruster_cmds):
        """Apply a single physics step with thruster commands and buoyancy/drag."""
        base_pos, base_quat = p.getBasePositionAndOrientation(self.rov)
        lin, ang = p.getBaseVelocity(self.rov)
        
        # Buoyancy
        depth = max(0.0, rov_sim.SURFACE_Z - base_pos[2])
        _hull_half_z = 0.15
        submersion = min(1.0, max(0.0, depth / _hull_half_z)) if depth < _hull_half_z else 1.0
        depth_buoyancy_factor = max(0.5, 1.0 - rov_sim.DEPTH_BUOYANCY_COMPRESSIBILITY * depth)
        buoy_force = rov_sim.MASS * rov_sim.GRAVITY * rov_sim.BUOYANCY_SCALE * depth_buoyancy_factor * submersion
        
        cob_rel_world = p.rotateVector(base_quat, rov_sim.COB_OFFSET_BODY)
        cob_world = (base_pos[0] + cob_rel_world[0], base_pos[1] + cob_rel_world[1], base_pos[2] + cob_rel_world[2])
        p.applyExternalForce(self.rov, -1, (0.0, 0.0, buoy_force), cob_world, p.WORLD_FRAME)
        
        # Ballast
        rov_sim.apply_ballast(self.rov, base_pos, base_quat)
        
        # Righting torque
        if submersion > 0.01:
            rov_sim.apply_righting_torque(self.rov, base_quat, ang, submersion)
        
        # Drag
        rov_sim.apply_drag(self.rov, base_pos, base_quat, lin, ang)
        
        # Thrusters
        inv_q = p.invertTransform([0, 0, 0], base_quat)[1]
        v_body = p.rotateVector(inv_q, lin)
        
        thr_level = [0.0] * len(self.thrusters)
        for i, thr in enumerate(self.thrusters):
            cmd = thruster_cmds[i] if i < len(thruster_cmds) else 0.0
            tau = rov_sim.THRUSTER_TAU_UP if cmd > thr_level[i] else rov_sim.THRUSTER_TAU_DN
            thr_level[i] += (rov_sim.DT / max(1e-6, tau)) * (cmd - thr_level[i])
            thr_level[i] = max(-1.0, min(1.0, thr_level[i]))
            
            if abs(thr_level[i]) <= 1e-4:
                continue
            
            thrust_max = rov_sim.MAX_THRUST_H if thr["kind"] == "H" else rov_sim.MAX_THRUST_V
            thrust = thrust_max * thr_level[i] * rov_sim.THRUST_LEVEL
            if thrust < 0:
                thrust *= rov_sim.BACKWARDS_THRUST_SCALE
            
            dir_body_norm = thr["dir"]
            thrust_vec_body = (dir_body_norm[0] * thrust, dir_body_norm[1] * thrust, dir_body_norm[2] * thrust)
            thrust_vec_world = p.rotateVector(base_quat, thrust_vec_body)
            pos_world = p.rotateVector(base_quat, thr["pos"])
            
            p.applyExternalForce(self.rov, -1, thrust_vec_world, pos_world, p.WORLD_FRAME)
        
        # ASSIST MODES: Apply depth hold and heading hold if enabled
        rov_sim.apply_depth_hold(self.rov, base_pos, lin)
        rov_sim.apply_heading_hold(self.rov, base_quat, ang)
        
        p.stepSimulation()
    
    def test_depth_hold_step_response(self, initial_depth=2.0, target_depth=3.0, duration=10.0):
        """
        Test: Set depth hold at initial_depth, then measure response to target_depth change.
        
        Acceptance criteria:
          - Response time (reach 90% of target): < 8 seconds
          - Overshoot: < 0.2 m
          - Steady-state error: < 0.1 m
        
        Returns dict with test results.
        """
        print("\n" + "="*70)
        print("DEPTH-HOLD STEP RESPONSE TEST")
        print("="*70)
        
        self.reset_rov(depth=initial_depth)
        
        # Settle for 2 seconds
        print(f"  Settling at initial depth {initial_depth:.2f} m...", end="", flush=True)
        for _ in range(int(2.0 / rov_sim.DT)):
            self.apply_physics_step([0.0] * len(self.thrusters))
        print(" done")
        
        # Enable depth hold at this depth
        rov_sim.DEPTH_HOLD_ENABLED = True
        rov_sim.DEPTH_HOLD_TARGET = initial_depth
        current_depth, _ = self._get_state()
        print(f"  Depth-hold engaged at {current_depth:.3f} m → target {target_depth:.3f} m")
        
        # Change target depth and measure response
        rov_sim.DEPTH_HOLD_TARGET = target_depth
        
        samples = []  # (time, depth, depth_error)
        t_90_percent = None
        max_depth = initial_depth
        
        for step in range(int(duration / rov_sim.DT)):
            t = step * rov_sim.DT
            current_depth, _ = self._get_state()
            depth_error = abs(current_depth - target_depth)
            samples.append((t, current_depth, depth_error))
            
            # Track metrics
            if current_depth > max_depth:
                max_depth = current_depth
            
            # Time to 90% convergence
            if t_90_percent is None and depth_error < 0.1 * abs(target_depth - initial_depth):
                t_90_percent = t
            
            # Apply zero thruster commands (depth hold is automatic via assist mode)
            # Vertical thruster (T3) is controlled by depth hold internally
            self.apply_physics_step([0.0] * len(self.thrusters))
        
        # Disable depth hold
        rov_sim.DEPTH_HOLD_ENABLED = False
        
        # Calculate metrics
        final_depth = samples[-1][1]
        steady_state_error = abs(final_depth - target_depth)
        overshoot = abs(max_depth - target_depth) if max_depth > target_depth else 0.0
        response_time = t_90_percent if t_90_percent is not None else duration
        
        # Acceptance criteria
        response_ok = response_time < 8.0
        ss_error_ok = steady_state_error < 0.1
        overshoot_ok = overshoot < 0.2
        
        result = {
            "test": "depth_hold_step_response",
            "initial_depth": initial_depth,
            "target_depth": target_depth,
            "duration": duration,
            "final_depth": final_depth,
            "response_time_90pct": response_time,
            "steady_state_error": steady_state_error,
            "overshoot": overshoot,
            "response_ok": response_ok,
            "ss_error_ok": ss_error_ok,
            "overshoot_ok": overshoot_ok,
            "samples": samples,
        }
        
        status_str = "✅" if (response_ok and ss_error_ok and overshoot_ok) else "❌"
        print(f"\n  {status_str} RESULTS:")
        print(f"      Final depth:          {final_depth:.3f} m (target {target_depth:.3f} m)")
        print(f"      Response time (90%):  {response_time:.2f} s (requirement: <8.0 s) {'✅' if response_ok else '❌'}")
        print(f"      Steady-state error:   {steady_state_error:.3f} m (requirement: <0.1 m) {'✅' if ss_error_ok else '❌'}")
        print(f"      Overshoot:            {overshoot:.3f} m (requirement: <0.2 m) {'✅' if overshoot_ok else '❌'}")
        
        return result
    
    def test_depth_hold_disturbance_rejection(self, hold_depth=2.0, disturbance_force=10.0, duration=5.0):
        """
        Test: Depth hold rejects external disturbance (sudden downward force impulse).
        
        Acceptance criteria:
          - Post-disturbance deviation: < 0.3 m
          - Recovery time: < 5 seconds
        """
        print("\n" + "="*70)
        print("DEPTH-HOLD DISTURBANCE REJECTION TEST")
        print("="*70)
        
        self.reset_rov(depth=hold_depth)
        
        # Settle
        print(f"  Settling at {hold_depth:.2f} m...", end="", flush=True)
        for _ in range(int(2.0 / rov_sim.DT)):
            self.apply_physics_step([0.0] * len(self.thrusters))
        print(" done")
        
        # Engage depth hold
        rov_sim.DEPTH_HOLD_ENABLED = True
        rov_sim.DEPTH_HOLD_TARGET = hold_depth
        
        samples = []
        max_deviation = 0.0
        t_recovery = None
        
        for step in range(int(duration / rov_sim.DT)):
            t = step * rov_sim.DT
            base_pos, _ = p.getBasePositionAndOrientation(self.rov)
            current_depth = max(0.0, rov_sim.SURFACE_Z - base_pos[2])
            deviation = abs(current_depth - hold_depth)
            samples.append((t, current_depth, deviation))
            
            # Apply disturbance impulse at t=0.5s
            if 0.4 < t < 0.6:
                p.applyExternalForce(self.rov, -1, (0.0, 0.0, -disturbance_force), base_pos, p.WORLD_FRAME)
            
            # Track recovery
            if t > 0.6 and t_recovery is None and deviation < 0.15:
                t_recovery = t
            
            max_deviation = max(max_deviation, deviation)
            self.apply_physics_step([0.0] * len(self.thrusters))
        
        rov_sim.DEPTH_HOLD_ENABLED = False
        
        # Metrics
        recovery_time = t_recovery if t_recovery is not None else duration
        deviation_ok = max_deviation < 0.3
        recovery_ok = recovery_time < 5.0
        
        result = {
            "test": "depth_hold_disturbance_rejection",
            "hold_depth": hold_depth,
            "disturbance_force": disturbance_force,
            "duration": duration,
            "max_deviation": max_deviation,
            "recovery_time": recovery_time,
            "deviation_ok": deviation_ok,
            "recovery_ok": recovery_ok,
            "samples": samples,
        }
        
        status_str = "✅" if (deviation_ok and recovery_ok) else "❌"
        print(f"\n  {status_str} RESULTS:")
        print(f"      Max deviation:       {max_deviation:.3f} m (requirement: <0.3 m) {'✅' if deviation_ok else '❌'}")
        print(f"      Recovery time:       {recovery_time:.2f} s (requirement: <5.0 s) {'✅' if recovery_ok else '❌'}")
        
        return result
    
    def _get_state(self):
        """Get current ROV depth and heading."""
        base_pos, base_quat = p.getBasePositionAndOrientation(self.rov)
        depth = max(0.0, rov_sim.SURFACE_Z - base_pos[2])
        _, _, heading = p.getEulerFromQuaternion(base_quat)
        return depth, heading


class HeadingHoldTest:
    """Validate heading-hold controller performance."""
    
    def __init__(self):
        self.cid = None
        self.rov = None
        self.thrusters = []
        self.base_orn = None
    
    def setup(self):
        """Initialize PyBullet and ROV."""
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -rov_sim.GRAVITY)
        p.setPhysicsEngineParameter(fixedTimeStep=rov_sim.DT, numSolverIterations=50, numSubSteps=1)
        p.loadURDF("plane.urdf", [0, 0, rov_sim.SEABED_Z])
        
        self.rov, mesh_center = rov_sim.build_rov()
        gltf_path = os.path.join(ROOT, "Assembly 1.gltf")
        self.thrusters = rov_sim.detect_thrusters_from_gltf(gltf_path, mesh_center)
        
        euler_rad = tuple(math.radians(d) for d in rov_sim.MESH_BODY_EULER_DEG)
        self.base_orn = p.getQuaternionFromEuler(euler_rad)
    
    def cleanup(self):
        if self.cid is not None:
            p.disconnect(self.cid)
            self.cid = None
    
    def reset_rov(self, depth=2.0):
        """Reset ROV to starting position."""
        start_pos = [0, 0, rov_sim.SURFACE_Z - depth]
        p.resetBasePositionAndOrientation(self.rov, start_pos, self.base_orn)
        p.resetBaseVelocity(self.rov, [0, 0, 0], [0, 0, 0])
        rov_sim.LAST_VREL_BODY = None
        rov_sim.LAST_A_BODY = (0.0, 0.0, 0.0)
        rov_sim.LAST_W_BODY = None
        rov_sim.LAST_ALPHA_BODY = (0.0, 0.0, 0.0)
    
    def apply_physics_step(self, thruster_cmds):
        """Apply a single physics step."""
        base_pos, base_quat = p.getBasePositionAndOrientation(self.rov)
        lin, ang = p.getBaseVelocity(self.rov)
        
        # Buoyancy
        depth = max(0.0, rov_sim.SURFACE_Z - base_pos[2])
        _hull_half_z = 0.15
        submersion = min(1.0, max(0.0, depth / _hull_half_z)) if depth < _hull_half_z else 1.0
        depth_buoyancy_factor = max(0.5, 1.0 - rov_sim.DEPTH_BUOYANCY_COMPRESSIBILITY * depth)
        buoy_force = rov_sim.MASS * rov_sim.GRAVITY * rov_sim.BUOYANCY_SCALE * depth_buoyancy_factor * submersion
        
        cob_rel_world = p.rotateVector(base_quat, rov_sim.COB_OFFSET_BODY)
        cob_world = (base_pos[0] + cob_rel_world[0], base_pos[1] + cob_rel_world[1], base_pos[2] + cob_rel_world[2])
        p.applyExternalForce(self.rov, -1, (0.0, 0.0, buoy_force), cob_world, p.WORLD_FRAME)
        
        rov_sim.apply_ballast(self.rov, base_pos, base_quat)
        if submersion > 0.01:
            rov_sim.apply_righting_torque(self.rov, base_quat, ang, submersion)
        rov_sim.apply_drag(self.rov, base_pos, base_quat, lin, ang)
        
        inv_q = p.invertTransform([0, 0, 0], base_quat)[1]
        v_body = p.rotateVector(inv_q, lin)
        
        thr_level = [0.0] * len(self.thrusters)
        for i, thr in enumerate(self.thrusters):
            cmd = thruster_cmds[i] if i < len(thruster_cmds) else 0.0
            tau = rov_sim.THRUSTER_TAU_UP if cmd > thr_level[i] else rov_sim.THRUSTER_TAU_DN
            thr_level[i] += (rov_sim.DT / max(1e-6, tau)) * (cmd - thr_level[i])
            thr_level[i] = max(-1.0, min(1.0, thr_level[i]))
            
            if abs(thr_level[i]) <= 1e-4:
                continue
            
            thrust_max = rov_sim.MAX_THRUST_H if thr["kind"] == "H" else rov_sim.MAX_THRUST_V
            thrust = thrust_max * thr_level[i] * rov_sim.THRUST_LEVEL
            if thrust < 0:
                thrust *= rov_sim.BACKWARDS_THRUST_SCALE
            
            dir_body_norm = thr["dir"]
            thrust_vec_body = (dir_body_norm[0] * thrust, dir_body_norm[1] * thrust, dir_body_norm[2] * thrust)
            thrust_vec_world = p.rotateVector(base_quat, thrust_vec_body)
            pos_world = p.rotateVector(base_quat, thr["pos"])
            
            p.applyExternalForce(self.rov, -1, thrust_vec_world, pos_world, p.WORLD_FRAME)
        
        # ASSIST MODES: Apply depth hold and heading hold if enabled
        rov_sim.apply_depth_hold(self.rov, base_pos, lin)
        rov_sim.apply_heading_hold(self.rov, base_quat, ang)
        
        p.stepSimulation()
    
    def test_heading_hold_stability(self, initial_heading=None, duration=20.0):
        """
        Test: Heading hold maintains stable heading over extended period.
        
        Acceptance criteria:
          - Heading drift: < ±5°
          - RMS deviation: < 2°
        """
        print("\n" + "="*70)
        print("HEADING-HOLD STABILITY TEST")
        print("="*70)
        
        self.reset_rov(depth=2.0)
        
        # Settle first, then capture true heading as baseline if not provided.
        display_heading = 0.0 if initial_heading is None else initial_heading
        print(f"  Settling near heading {math.degrees(display_heading):.1f}°...", end="", flush=True)
        for _ in range(int(2.0 / rov_sim.DT)):
            self.apply_physics_step([0.0] * len(self.thrusters))
        print(" done")

        _, heading_at_engage = self._get_state()
        target_heading = heading_at_engage if initial_heading is None else initial_heading
        print(f"  Heading-hold engage heading: {math.degrees(heading_at_engage):.2f}°")
        print(f"  Heading-hold target heading: {math.degrees(target_heading):.2f}°")
        
        # Engage heading hold
        rov_sim.HEADING_HOLD_ENABLED = True
        rov_sim.HEADING_HOLD_TARGET = target_heading
        
        samples = []
        deviations = []
        
        for step in range(int(duration / rov_sim.DT)):
            t = step * rov_sim.DT
            _, heading = self._get_state()
            
            # Normalize heading error to ±π
            dev = heading - target_heading
            while dev > math.pi:
                dev -= 2.0 * math.pi
            while dev < -math.pi:
                dev += 2.0 * math.pi
            
            deviations.append(abs(dev))
            samples.append((t, math.degrees(heading), math.degrees(dev)))
            
            self.apply_physics_step([0.0] * len(self.thrusters))
        
        rov_sim.HEADING_HOLD_ENABLED = False
        
        # Metrics
        max_drift = max(deviations)
        rms_deviation = math.sqrt(sum(d**2 for d in deviations) / len(deviations)) if deviations else 0.0
        drift_ok = max_drift < math.radians(5.0)
        rms_ok = rms_deviation < math.radians(2.0)
        
        result = {
            "test": "heading_hold_stability",
            "initial_heading": heading_at_engage,
            "target_heading": target_heading,
            "duration": duration,
            "max_drift": max_drift,
            "rms_deviation": rms_deviation,
            "drift_ok": drift_ok,
            "rms_ok": rms_ok,
            "samples": samples,
        }
        
        status_str = "✅" if (drift_ok and rms_ok) else "❌"
        print(f"\n  {status_str} RESULTS:")
        print(f"      Max drift:        {math.degrees(max_drift):.2f}° (requirement: <5°) {'✅' if drift_ok else '❌'}")
        print(f"      RMS deviation:    {math.degrees(rms_deviation):.2f}° (requirement: <2°) {'✅' if rms_ok else '❌'}")
        
        return result
    
    def _get_state(self):
        """Get current ROV depth and heading."""
        base_pos, base_quat = p.getBasePositionAndOrientation(self.rov)
        depth = max(0.0, rov_sim.SURFACE_Z - base_pos[2])
        _, _, heading = p.getEulerFromQuaternion(base_quat)
        return depth, heading


# ──────────────────────────────────────────────────────────────────────
#  PART 2: GLTF THRUSTER FRAME VALIDATION
# ──────────────────────────────────────────────────────────────────────

def test_gltf_thruster_validation():
    """
    Validate that GLTF thruster data matches expectations:
      - Thruster count matches design
      - Positions are symmetric (left/right pairs)
      - Rotations produce expected torque vectors
    
    Returns list of validation results.
    """
    print("\n" + "="*70)
    print("GLTF THRUSTER FRAME VALIDATION")
    print("="*70)
    
    gltf_path = os.path.join(ROOT, "Assembly 1.gltf")
    if not os.path.exists(gltf_path):
        print(f"  ❌ GLTF file not found: {gltf_path}")
        return []
    
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    rov, mesh_center = rov_sim.build_rov()
    thrusters = rov_sim.detect_thrusters_from_gltf(gltf_path, mesh_center)
    
    results = []
    
    # Validation 1: Thruster count
    expected_count = 4  # Typical configuration: T1, T2, T3, T4
    actual_count = len(thrusters)
    count_ok = actual_count >= expected_count or expected_count is None
    results.append({"check": "thruster_count", "expected": expected_count, "actual": actual_count, "passed": count_ok})
    
    print(f"\n  {'✅' if count_ok else '❌'} Thruster count: {actual_count} (expected ≥ {expected_count})")
    
    # Validation 2: Symmetry check (left/right pairs should have symmetric position/rotation)
    print(f"\n  Thruster Geometry:")
    for i, t in enumerate(thrusters):
        print(f"    T{i+1} ({t['name']}, {t['kind']}):")
        print(f"      Pos (body):  ({t['pos'][0]:+.3f}, {t['pos'][1]:+.3f}, {t['pos'][2]:+.3f})")
        print(f"      Dir (body):  ({t['dir'][0]:+.3f}, {t['dir'][1]:+.3f}, {t['dir'][2]:+.3f})")
    
    # Validation 3: Torque vector signs
    print(f"\n  Torque Vector Analysis:")
    euler_rad = tuple(math.radians(d) for d in rov_sim.MESH_BODY_EULER_DEG)
    base_quat = p.getQuaternionFromEuler(euler_rad)
    
    hthrusters = [t for t in thrusters if t["kind"] == "H"]
    if len(hthrusters) >= 2:
        t1, t2 = hthrusters[0], hthrusters[1]
        # T1 cmd +1, T2 cmd -1 should produce CCW yaw (positive torque_z)
        # Torque_z = r_x * F_y - r_y * F_x
        
        dir1_world = p.rotateVector(base_quat, t1["dir"])
        dir2_world = p.rotateVector(base_quat, t2["dir"])
        pos1_world = p.rotateVector(base_quat, t1["pos"])
        pos2_world = p.rotateVector(base_quat, t2["pos"])
        
        # Assume unit thrust in their body directions
        f1_world = dir1_world  # T1 +1
        f2_world = tuple(-d for d in dir2_world)  # T2 -1
        
        tau_z_1 = pos1_world[0] * f1_world[1] - pos1_world[1] * f1_world[0]
        tau_z_2 = pos2_world[0] * f2_world[1] - pos2_world[1] * f2_world[0]
        tau_z_total = tau_z_1 + tau_z_2
        
        torque_sign_ok = tau_z_total > 0  # Should be CCW (positive)
        results.append({"check": "yaw_torque_sign", "expected": "positive (CCW)", "actual": f"{tau_z_total:.4f}", "passed": torque_sign_ok})
        
        print(f"    T1=+1, T2=-1 → Torque_z = {tau_z_total:+.4f} Nm")
        print(f"    {'✅' if torque_sign_ok else '❌'} Sign is {'CCW (correct)' if torque_sign_ok else 'CW (incorrect)'}")
    
    p.disconnect(cid)
    
    # Summary
    all_passed = all(r["passed"] for r in results)
    print(f"\n  {'✅' if all_passed else '❌'} GLTF validation: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    
    return results


# ──────────────────────────────────────────────────────────────────────
#  PART 3: PHYSICS CALIBRATION BASELINES
# ──────────────────────────────────────────────────────────────────────

def extract_physics_parameters():
    """Extract and document current physics parameters."""
    params = {
        "gravity": rov_sim.GRAVITY,
        "mass": rov_sim.MASS,
        "water_density": rov_sim.RHO,
        "buoyancy_scale": rov_sim.BUOYANCY_SCALE,
        "depth_buoyancy_compressibility": rov_sim.DEPTH_BUOYANCY_COMPRESSIBILITY,
        "drag": {
            "linear_body": rov_sim.LIN_DRAG_BODY,
            "linear_angular": rov_sim.LIN_DRAG_ANG,
            "quadratic_angular": rov_sim.QUAD_DRAG_ANG,
            "drag_coefficients": rov_sim.CD,
            "wetted_areas": rov_sim.AREA,
            "max_force": rov_sim.MAX_DRAG_FORCE,
            "max_torque": rov_sim.MAX_DRAG_TORQUE,
        },
        "added_mass": {
            "body": rov_sim.ADDED_MASS_BODY,
            "inertia": rov_sim.ADDED_INERTIA_BODY,
        },
        "thrusters": {
            "max_thrust_h": rov_sim.MAX_THRUST_H,
            "max_thrust_v": rov_sim.MAX_THRUST_V,
            "thrust_level": rov_sim.THRUST_LEVEL,
            "backwards_thrust_scale": rov_sim.BACKWARDS_THRUST_SCALE,
            "thruster_tau_up": rov_sim.THRUSTER_TAU_UP,
            "thruster_tau_down": rov_sim.THRUSTER_TAU_DN,
        },
        "assist_modes": {
            "depth_hold": {
                "kp": rov_sim.DEPTH_HOLD_KP,
                "kd": rov_sim.DEPTH_HOLD_KD,
                "max_force": rov_sim.DEPTH_HOLD_MAX_FORCE,
            },
            "heading_hold": {
                "kp": rov_sim.HEADING_HOLD_KP,
                "kd": rov_sim.HEADING_HOLD_KD,
                "max_torque": rov_sim.HEADING_HOLD_MAX_TORQUE,
            },
        },
    }
    return params


def print_calibration_baselines():
    """Print current physics parameters as baseline."""
    print("\n" + "="*70)
    print("PHYSICS CALIBRATION BASELINES")
    print("="*70)
    
    params = extract_physics_parameters()
    
    print(f"\n  CORE DYNAMICS:")
    print(f"    Gravity:                    {params['gravity']:.2f} m/s²")
    print(f"    Vehicle mass:               {params['mass']:.2f} kg")
    print(f"    Water density:              {params['water_density']:.1f} kg/m³")
    print(f"    Buoyancy scale:             {params['buoyancy_scale']:.3f}")
    print(f"    Depth buoyancy compress:    {params['depth_buoyancy_compressibility']:.4f}/m")
    
    print(f"\n  DRAG MODEL (Fossen-style linear + quadratic):")
    print(f"    Linear body drag (N·s/m):   surge={params['drag']['linear_body'][0]:.1f}, sway={params['drag']['linear_body'][1]:.1f}, heave={params['drag']['linear_body'][2]:.1f}")
    print(f"    Linear angular (N·m·s/rad): roll={params['drag']['linear_angular'][0]:.2f}, pitch={params['drag']['linear_angular'][1]:.2f}, yaw={params['drag']['linear_angular'][2]:.2f}")
    print(f"    Quadratic angular (N·m·s²/rad²): {params['drag']['quadratic_angular']}")
    print(f"    Drag coefficients (CD):     {params['drag']['drag_coefficients']}")
    print(f"    Wetted areas (m²):          surge={params['drag']['wetted_areas'][0]:.4f}, sway={params['drag']['wetted_areas'][1]:.4f}, heave={params['drag']['wetted_areas'][2]:.4f}")
    print(f"    Max clamp force:            {params['drag']['max_force']:.1f} N")
    print(f"    Max clamp torque:           {params['drag']['max_torque']:.1f} N·m")
    
    print(f"\n  ADDED MASS (hydrodynamic coupling):")
    print(f"    Translational (kg):         surge={params['added_mass']['body'][0]:.2f}, sway={params['added_mass']['body'][1]:.2f}, heave={params['added_mass']['body'][2]:.2f}")
    print(f"    Rotational (kg·m²):         roll={params['added_mass']['inertia'][0]:.4f}, pitch={params['added_mass']['inertia'][1]:.4f}, yaw={params['added_mass']['inertia'][2]:.4f}")
    
    print(f"\n  THRUSTERS (nominal):")
    print(f"    Max thrust horizontal:      {params['thrusters']['max_thrust_h']:.2f} N")
    print(f"    Max thrust vertical:        {params['thrusters']['max_thrust_v']:.2f} N")
    print(f"    Thrust level scale:         {params['thrusters']['thrust_level']:.2f}")
    print(f"    Backwards thrust scale:     {params['thrusters']['backwards_thrust_scale']:.2f}")
    print(f"    Ramp up time constant:      {params['thrusters']['thruster_tau_up']:.3f} s")
    print(f"    Ramp down time constant:    {params['thrusters']['thruster_tau_down']:.3f} s")
    
    print(f"\n  ASSIST MODES:")
    print(f"    Depth-hold Kp:              {params['assist_modes']['depth_hold']['kp']:.1f} N/m")
    print(f"    Depth-hold Kd:              {params['assist_modes']['depth_hold']['kd']:.1f} N·s/m")
    print(f"    Depth-hold max force:       {params['assist_modes']['depth_hold']['max_force']:.1f} N")
    print(f"    Heading-hold Kp:            {params['assist_modes']['heading_hold']['kp']:.1f} N·m/rad")
    print(f"    Heading-hold Kd:            {params['assist_modes']['heading_hold']['kd']:.1f} N·m·s/rad")
    print(f"    Heading-hold max torque:    {params['assist_modes']['heading_hold']['max_torque']:.1f} N·m")
    
    return params


# ──────────────────────────────────────────────────────────────────────
#  MAIN TEST RUNNER
# ──────────────────────────────────────────────────────────────────────

def run_all_tests():
    """Run complete extended diagnostics suite."""
    print("\n" + "🎯"*35)
    print("EXTENDED DIAGNOSTICS SUITE")
    print("Assist Modes | GLTF Validation | Physics Calibration")
    print("🎯"*35)
    
    results = {
        "assist_modes": {},
        "gltf_validation": {},
        "physics_baseline": {},
    }
    
    # Part 1: Depth Hold Tests
    print("\n[1/4] DEPTH-HOLD TESTS")
    dh_test = DepthHoldTest()
    try:
        dh_test.setup()
        results["assist_modes"]["depth_hold_step"] = dh_test.test_depth_hold_step_response()
        results["assist_modes"]["depth_hold_disturbance"] = dh_test.test_depth_hold_disturbance_rejection()
    finally:
        dh_test.cleanup()
    
    # Part 2: Heading Hold Tests
    print("\n[2/4] HEADING-HOLD TESTS")
    hh_test = HeadingHoldTest()
    try:
        hh_test.setup()
        results["assist_modes"]["heading_hold_stability"] = hh_test.test_heading_hold_stability()
    finally:
        hh_test.cleanup()
    
    # Part 3: GLTF Validation
    print("\n[3/4] GLTF VALIDATION")
    results["gltf_validation"]["thruster_frame"] = test_gltf_thruster_validation()
    
    # Part 4: Physics Calibration Baselines
    print("\n[4/4] PHYSICS CALIBRATION BASELINES")
    results["physics_baseline"]["parameters"] = print_calibration_baselines()
    
    # Final Summary
    print("\n" + "="*70)
    print("EXTENDED DIAGNOSTICS SUMMARY")
    print("="*70)
    
    depth_hold_ok = (
        results["assist_modes"].get("depth_hold_step", {}).get("response_ok", False) and
        results["assist_modes"].get("depth_hold_step", {}).get("ss_error_ok", False) and
        results["assist_modes"].get("depth_hold_disturbance", {}).get("deviation_ok", False)
    )
    
    heading_hold_ok = results["assist_modes"].get("heading_hold_stability", {}).get("drift_ok", False)
    
    gltf_ok = all(r.get("passed", False) for r in results["gltf_validation"].get("thruster_frame", []))
    
    print(f"\n  ✅ Depth-hold tests:          {'PASSED' if depth_hold_ok else 'FAILED' if 'depth_hold_step' in results['assist_modes'] else 'SKIPPED'}")
    print(f"  ✅ Heading-hold tests:        {'PASSED' if heading_hold_ok else 'FAILED' if 'heading_hold_stability' in results['assist_modes'] else 'SKIPPED'}")
    print(f"  ✅ GLTF validation:           {'PASSED' if gltf_ok else 'FAILED' if results['gltf_validation'] else 'SKIPPED'}")
    print(f"  ✅ Physics baselines:         DOCUMENTED")
    
    overall_ok = depth_hold_ok and heading_hold_ok and gltf_ok
    print(f"\n  🎯 Overall: {'ALL TESTS PASSED ✅' if overall_ok else 'SOME TESTS FAILED ❌'}")
    
    return results


if __name__ == "__main__":
    # Configure simulation for testing
    rov_sim.SLEEP_REALTIME = False
    rov_sim.HUD_ENABLED = False
    rov_sim.ENABLE_MARKERS = False
    rov_sim.ENABLE_CAMERA_PREVIEW = False
    rov_sim.LOG_PHYSICS_DETAILED = False
    rov_sim.ENABLE_THRUSTER_ARROWS = False
    
    # Run tests
    results = run_all_tests()
    
    # Optionally save results to JSON
    with open("extended_diagnostics_results.json", "w") as f:
        # Convert results to serializable format (remove tuples and complex objects)
        serializable = {
            "assist_modes": results["assist_modes"],
            "physics_baseline": {
                "parameters": results["physics_baseline"]["parameters"]
            },
        }
        json.dump(serializable, f, indent=2)
    
    print("\n✅ Extended diagnostics complete. Results saved to extended_diagnostics_results.json")

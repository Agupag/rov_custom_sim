#!/usr/bin/env python3
"""
Quick diagnostic: trace the full sign chain from joystick drag → yaw → thruster → torque.
Also compute optimal mixer coefficients for all directions.
"""
import os, sys, math
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pybullet as p
import pybullet_data
import rov_sim
import joystick_panel as jp
from joystick_panel import mix_joystick_to_thruster_cmds

def vmag(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def vcross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def vdot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

# ── Setup ──
cid = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -rov_sim.GRAVITY)
p.setPhysicsEngineParameter(fixedTimeStep=rov_sim.DT, numSolverIterations=50, numSubSteps=1)
p.loadURDF("plane.urdf", [0, 0, rov_sim.SEABED_Z])

rov_sim.SLEEP_REALTIME = False
rov_sim.HUD_ENABLED = False
rov_sim.ENABLE_MARKERS = False
rov_sim.ENABLE_CAMERA_PREVIEW = False
rov_sim.ENABLE_THRUSTER_ARROWS = False

rov, mesh_center = rov_sim.build_rov()
auto_thr = rov_sim.detect_thrusters_from_gltf(rov_sim.GLTF_FILE, mesh_center)
if auto_thr:
    rov_sim.THRUSTERS = auto_thr
THRUSTERS = rov_sim.THRUSTERS

base_pos, base_quat = p.getBasePositionAndOrientation(rov)

print("=" * 72)
print("  THRUSTER GEOMETRY ANALYSIS")
print("=" * 72)

# Camera forward in world frame
cam_fwd_body = (0.0, -1.0, 0.0)  # mesh frame forward
cam_fwd_world = p.rotateVector(base_quat, cam_fwd_body)
print(f"\n  Camera forward (body):  {cam_fwd_body}")
print(f"  Camera forward (world): ({cam_fwd_world[0]:.4f}, {cam_fwd_world[1]:.4f}, {cam_fwd_world[2]:.4f})")
print(f"  Body rotation: {rov_sim.MESH_BODY_EULER_DEG}")

print(f"\n  {'Thr':>4s}  {'Kind':>4s}  {'Pos(body)':>30s}  {'Dir(body)':>30s}  {'Dir(world)':>30s}  {'Fwd_comp':>8s}")
for i, t in enumerate(THRUSTERS):
    dw = p.rotateVector(base_quat, t["dir"])
    fwd_comp = vdot(dw, cam_fwd_world) / max(1e-9, vmag(cam_fwd_world))
    print(f"  T{i+1:>2d}    {t['kind']:>4s}  ({t['pos'][0]:+.4f}, {t['pos'][1]:+.4f}, {t['pos'][2]:+.4f})  "
          f"({t['dir'][0]:+.4f}, {t['dir'][1]:+.4f}, {t['dir'][2]:+.4f})  "
          f"({dw[0]:+.4f}, {dw[1]:+.4f}, {dw[2]:+.4f})  {fwd_comp:+.4f}")

# ── Analyze what each thruster contributes to each DOF ──
print(f"\n{'─'*72}")
print("  THRUSTER FORCE/TORQUE DECOMPOSITION (world frame)")
print(f"{'─'*72}")
print(f"  For cmd=+1.0 on each thruster (MAX_THRUST_H = {rov_sim.MAX_THRUST_H} N):")

for i, t in enumerate(THRUSTERS):
    if t["kind"] == "V":
        continue
    dw = p.rotateVector(base_quat, t["dir"])
    F = rov_sim.MAX_THRUST_H
    force = (dw[0]*F, dw[1]*F, dw[2]*F)
    
    # Force along camera-forward (world +X after 90° rotation)
    fwd_force = vdot(force, (cam_fwd_world[0]/vmag(cam_fwd_world), cam_fwd_world[1]/vmag(cam_fwd_world), cam_fwd_world[2]/vmag(cam_fwd_world)))
    
    # Lateral force (perpendicular to forward in XY plane)
    lat_dir = (-cam_fwd_world[1]/vmag(cam_fwd_world), cam_fwd_world[0]/vmag(cam_fwd_world), 0)
    lat_force = vdot(force, lat_dir)
    
    # Torque about COM
    rel_world = p.rotateVector(base_quat, t["pos"])
    torque = vcross(rel_world, force)
    
    print(f"\n  T{i+1} (cmd=+1):")
    print(f"    Force (world): ({force[0]:+.3f}, {force[1]:+.3f}, {force[2]:+.3f}) N")
    print(f"    Forward component: {fwd_force:+.3f} N")
    print(f"    Lateral component: {lat_force:+.3f} N  (+ = starboard/right)")
    print(f"    Torque about COM:  ({torque[0]:+.4f}, {torque[1]:+.4f}, {torque[2]:+.4f}) N·m")
    print(f"    Yaw torque (τz):   {torque[2]:+.4f} N·m  ({'CW/right' if torque[2] < 0 else 'CCW/left'})")

# ── Compute optimal mixer for each direction ──
print(f"\n{'─'*72}")
print("  OPTIMAL MIXER ANALYSIS")
print(f"{'─'*72}")
print("  For each desired direction, what combination of T1,T2,T4 maximizes thrust?")

# Get horizontal thruster data
h_thrusters = []
for i, t in enumerate(THRUSTERS):
    if t["kind"] == "V":
        continue
    dw = p.rotateVector(base_quat, t["dir"])
    fwd = vdot(dw, (cam_fwd_world[0]/vmag(cam_fwd_world), cam_fwd_world[1]/vmag(cam_fwd_world), cam_fwd_world[2]/vmag(cam_fwd_world)))
    lat_dir = (-cam_fwd_world[1]/vmag(cam_fwd_world), cam_fwd_world[0]/vmag(cam_fwd_world), 0)
    lat = vdot(dw, lat_dir)
    rel_world = p.rotateVector(base_quat, t["pos"])
    torque_z = vcross(rel_world, dw)[2]  # per unit thrust
    h_thrusters.append({'idx': i, 'name': t['name'], 'fwd': fwd, 'lat': lat, 'tz': torque_z})

print(f"\n  Per-thruster unit force components:")
for h in h_thrusters:
    print(f"    T{h['idx']+1}: fwd={h['fwd']:+.4f}  lat={h['lat']:+.4f}  yaw_torque={h['tz']:+.4f}")

# For each direction, try all 3^3 = 27 combinations of {-1, 0, +1} for T1,T2,T4
directions = {
    'Forward':     {'fwd':  1.0, 'lat': 0.0, 'yaw': 0.0},
    'Reverse':     {'fwd': -1.0, 'lat': 0.0, 'yaw': 0.0},
    'Yaw Right':   {'fwd':  0.0, 'lat': 0.0, 'yaw': -1.0},
    'Yaw Left':    {'fwd':  0.0, 'lat': 0.0, 'yaw':  1.0},
    'Fwd+YawR':    {'fwd':  1.0, 'lat': 0.0, 'yaw': -1.0},
    'Fwd+YawL':    {'fwd':  1.0, 'lat': 0.0, 'yaw':  1.0},
    'Rev+YawR':    {'fwd': -1.0, 'lat': 0.0, 'yaw': -1.0},
    'Rev+YawL':    {'fwd': -1.0, 'lat': 0.0, 'yaw':  1.0},
}

print(f"\n  OPTIMAL COMMANDS per direction (maximizing desired DOF, minimizing unwanted):")
for name, goal in directions.items():
    best_score = -1e9
    best_cmds = None
    
    for t1 in [-1, 0, 1]:
        for t2 in [-1, 0, 1]:
            for t4 in [-1, 0, 1]:
                net_fwd = t1*h_thrusters[0]['fwd'] + t2*h_thrusters[1]['fwd'] + t4*h_thrusters[2]['fwd']
                net_lat = t1*h_thrusters[0]['lat'] + t2*h_thrusters[1]['lat'] + t4*h_thrusters[2]['lat']
                net_yaw = t1*h_thrusters[0]['tz']  + t2*h_thrusters[1]['tz']  + t4*h_thrusters[2]['tz']
                
                # Score: reward desired DOFs, penalize unwanted
                score = 0
                if abs(goal['fwd']) > 0.1:
                    score += net_fwd * goal['fwd'] * 5.0  # strongly reward forward/reverse alignment
                else:
                    score -= abs(net_fwd) * 2.0  # penalize unwanted translation
                    
                if abs(goal['yaw']) > 0.1:
                    score += net_yaw * goal['yaw'] * 5.0  # reward yaw alignment
                else:
                    score -= abs(net_yaw) * 2.0  # penalize unwanted yaw
                
                score -= abs(net_lat) * 1.0  # always penalize lateral drift
                
                if score > best_score:
                    best_score = score
                    best_cmds = (t1, t2, t4)
                    best_fwd = net_fwd
                    best_lat = net_lat
                    best_yaw = net_yaw
    
    print(f"\n  {name:>12s}: T1={best_cmds[0]:+d}  T2={best_cmds[1]:+d}  T4={best_cmds[2]:+d}")
    print(f"                Net fwd={best_fwd*rov_sim.MAX_THRUST_H:+.2f}N  lat={best_lat*rov_sim.MAX_THRUST_H:+.2f}N  yaw_τ={best_yaw*rov_sim.MAX_THRUST_H:+.4f}N·m")

# ── Current mixer check ──
print(f"\n{'─'*72}")
print("  CURRENT MIXER VS OPTIMAL")
print(f"{'─'*72}")

mixer_tests = [
    ("Forward",    {"surge": 1.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}),
    ("Reverse",    {"surge":-1.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}),
    ("Yaw Right",  {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 1.0}),
    ("Yaw Left",   {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw":-1.0}),
    ("Fwd+YawR",   {"surge": 0.8, "sway": 0.0, "heave": 0.0, "yaw": 0.5}),
    ("Fwd+YawL",   {"surge": 0.8, "sway": 0.0, "heave": 0.0, "yaw":-0.5}),
]

for label, state in mixer_tests:
    cmds = mix_joystick_to_thruster_cmds(state, 4)
    print(f"  {label:>12s}: joystick(surge={state['surge']:+.1f}, yaw={state['yaw']:+.1f}) → T1={cmds[0]:+.1f} T2={cmds[1]:+.1f} T3={cmds[2]:+.1f} T4={cmds[3]:+.1f}")

# ── Yaw sign check ──
print(f"\n{'─'*72}")
print("  YAW SIGN CHAIN TRACE")
print(f"{'─'*72}")
print("  User drags joystick LEFT on screen:")
print("    → knob moves left → dx < 0 → ax = dx/radius < 0")
print("    → _set(YAW, ax) → yaw < 0 (NEGATIVE)")
cmds_yaw_neg = mix_joystick_to_thruster_cmds({"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": -1.0}, 4)
print(f"    → T1={cmds_yaw_neg[0]:+.1f} T2={cmds_yaw_neg[1]:+.1f} T4={cmds_yaw_neg[3]:+.1f}")

net_tz = 0
for i, t in enumerate(THRUSTERS):
    if t["kind"] == "V" or cmds_yaw_neg[i] == 0:
        continue
    dw = p.rotateVector(base_quat, t["dir"])
    force = (dw[0]*rov_sim.MAX_THRUST_H*cmds_yaw_neg[i], dw[1]*rov_sim.MAX_THRUST_H*cmds_yaw_neg[i], dw[2]*rov_sim.MAX_THRUST_H*cmds_yaw_neg[i])
    rel_world = p.rotateVector(base_quat, t["pos"])
    torque = vcross(rel_world, force)
    net_tz += torque[2]
print(f"    → Net τz = {net_tz:+.4f} N·m → {'CCW (LEFT from above)' if net_tz > 0 else 'CW (RIGHT from above)'}")
print(f"    → User dragged LEFT → ROV should yaw LEFT → τz should be POSITIVE (CCW)")
print(f"    → {'✅ CORRECT' if net_tz > 0 else '❌ BACKWARDS — need to fix yaw sign'}")

print()
print("  User drags joystick RIGHT on screen:")
print("    → knob moves right → dx > 0 → ax = dx/radius > 0")
print("    → _set(YAW, ax) → yaw > 0 (POSITIVE)")
cmds_yaw_pos = mix_joystick_to_thruster_cmds({"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 1.0}, 4)
print(f"    → T1={cmds_yaw_pos[0]:+.1f} T2={cmds_yaw_pos[1]:+.1f} T4={cmds_yaw_pos[3]:+.1f}")

net_tz2 = 0
for i, t in enumerate(THRUSTERS):
    if t["kind"] == "V" or cmds_yaw_pos[i] == 0:
        continue
    dw = p.rotateVector(base_quat, t["dir"])
    force = (dw[0]*rov_sim.MAX_THRUST_H*cmds_yaw_pos[i], dw[1]*rov_sim.MAX_THRUST_H*cmds_yaw_pos[i], dw[2]*rov_sim.MAX_THRUST_H*cmds_yaw_pos[i])
    rel_world = p.rotateVector(base_quat, t["pos"])
    torque = vcross(rel_world, force)
    net_tz2 += torque[2]
print(f"    → Net τz = {net_tz2:+.4f} N·m → {'CCW (LEFT from above)' if net_tz2 > 0 else 'CW (RIGHT from above)'}")
print(f"    → User dragged RIGHT → ROV should yaw RIGHT → τz should be NEGATIVE (CW)")
print(f"    → {'✅ CORRECT' if net_tz2 < 0 else '❌ BACKWARDS — need to fix yaw sign'}")

p.disconnect()

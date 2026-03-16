import os
import time
import math
import json
import random
from datetime import datetime

try:
    import cv2  # optional (used for camera preview window)
except Exception:
    cv2 = None

import pybullet as p
import pybullet_data

# If the simulator is asked to quit via ESC, set this flag so an external
# restart loop does not immediately relaunch the simulator.
USER_QUIT = False

# ==========================
# FILE LOGGING SETUP
# ==========================
HERE = os.path.dirname(__file__)
LOG_FILE = os.path.join(HERE, f"rov_sim_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Store original print and create logged version
_original_print = print
_log_file_handle = None

def print(*args, **kwargs):
    """Print to both console and log file."""
    global _log_file_handle
    msg = " ".join(str(a) for a in args)
    _original_print(*args, **kwargs)
    try:
        if _log_file_handle is not None:
            _log_file_handle.write(msg + "\n")
            _log_file_handle.flush()
    except Exception:
        pass

"""
Underwater ROV sim (PyBullet) — GLTF-aligned thrusters

- Loads your CAD visual mesh (Assembly_1.obj) and uses a stable box collision.
- Neutral buoyancy (buoyancy force = weight).
- Quadratic drag + rotational damping (simple underwater model).
- 4 thrusters toggled by keys 1-4 (ON/OFF).
- Thruster markers: RED=off, GREEN=on, placed at thruster positions and aligned to thruster direction.
- Thruster placement + direction are taken from Assembly_1.gltf if present (nodes: Thruster <1..4>).

Camera:
- Mouse works normally.
- Keyboard: J/L yaw, I/K pitch, U/O zoom, N/M move target up/down.
- R resets, ESC quits.

Notes:
- Ensure Assembly_1.obj and Assembly_1.gltf are in the same folder as this script.
"""

# ==========================
# CONFIG
# ==========================
DT = 1/90  # 90 Hz (optimized for Mac + physics stability)
SLEEP_REALTIME = True

GRAVITY = 9.81
MASS = 12.0

# ============================================
# HYDROSTATICS: Buoyancy + Ballast for Self-Righting
# ============================================
COB_OFFSET_BODY = (0.0, 0.0, 0.35)   # meters above COM (strong righting torque)
BUOYANCY_SCALE  = 1.12               # buoyancy force = 1.12 * weight (slightly positive)

BALLAST_OFFSET_BODY = (0.0, 0.0, -0.30)  # meters below COM (bottom-heavy)
BALLAST_SCALE       = 0.12               # extra downward force = 0.12 * weight

# PD controller for roll/pitch righting (avoids oscillation)
RIGHTING_K_RP = 35.0   # N*m per rad (proportional term)
RIGHTING_KD_RP = 15.0  # N*m per (rad/s) (derivative damping) — increased for stability

# Water current in WORLD frame (m/s). Set to (0,0,0) to disable.
WATER_CURRENT_WORLD = (0.00, 0.00, 0.00)

# ============================================
# THRUST CONFIGURATION
# ============================================
MAX_THRUST_H = 30.0  # Increased from 26.0 for better control authority (+15%)
MAX_THRUST_V = 20.0  # Reduced from 38

# Asymmetric thruster ramping (realistic motor dynamics)
THRUSTER_TAU_UP = 0.18  # Faster ramp-up
THRUSTER_TAU_DN = 0.08  # Slower ramp-down (creep behavior)

# ============================================
# HYDRODYNAMIC DRAG MODEL (Improved)
# ============================================
RHO = 1000.0  # Water density (kg/m^3)

# Quadratic drag coefficients (tuned for ROV shape)
CD = [0.22, 0.28, 0.48]
AREA = [0.055, 0.070, 0.060]

# Linear drag in BODY frame (N per m/s) for low-speed damping
LIN_DRAG_BODY = (10.1, 13.3, 17.3)  # Optimized from (15.4, 20.3, 26.4): reduced by 1.52x more (achieved 1.97 m/s, target 3.0 m/s)

# Disable added-mass and thruster inflow loss (restore lightweight default behavior).
ADDED_MASS_BODY = (0.0, 0.0, 0.0)
THRUSTER_SPEED_LOSS_COEF = 0.0

# Added-mass accel filter (not used when ADDED_MASS_BODY is zero)
ACCEL_FILTER_ALPHA = 0.0

# Internal state: remember previous velocity relative to water in BODY frame so we can
# estimate body acceleration (used for added-mass force). Initialized to zero.
LAST_VREL_BODY = (0.0, 0.0, 0.0)
# Low-pass filtered acceleration estimate (BODY frame)
LAST_A_BODY = (0.0, 0.0, 0.0)

# Rotational damping (WORLD frame baseline)
ROT_DAMP = 2.2

# Body-frame angular damping (roll, pitch, yaw)
ANG_DAMP_BODY = (9.0, 9.0, 3.2)

# ============================================
# STABILITY CLAMPS
# ============================================
MAX_SPEED = 6.0   # Reduced from 12 for safer, more realistic underwater behavior
MAX_OMEGA = 6.0   # Reduced from 12
MAX_DRAG_FORCE = 400.0  # Clamp on linear drag forces (N)
MAX_DRAG_TORQUE = 150.0  # Clamp on damping torques (N*m)

# Mesh files
OBJ_FILE = os.path.join(HERE, "Assembly 1.obj")
GLTF_FILE = os.path.join(HERE, "Assembly 1.gltf")

MESH_SCALE = (1.0, 1.0, 1.0)

# Rotate body so the mesh faces the right direction in the sim.
# If forward/backward looks flipped, try +90 instead of -90.
MESH_BODY_EULER_DEG = (0.0, 0.0, 90.0)  # Face forward towards obstacles

AUTO_DETECT_THRUSTERS = True  # uses GLTF nodes if available

# ============================================
# RENDERING OPTIMIZATION
# ============================================
ENABLE_CAMERA_PREVIEW = False  # Disabled by default (CPU intensive)
ENABLE_THRUSTER_ARROWS = True  # Visual thrust direction indicators
ENABLE_MARKERS = False  # Thruster position marker spheres (expensive)

# Camera defaults + keyboard steps
# Positioned to view ROV and obstacles in front of it
CAM_DIST = 2.2      # Slightly further back for better overview
CAM_YAW = 0         # Look straight at ROV from ahead
CAM_PITCH = -30     # Look slightly down at ROV
CAM_TARGET = [0.3, 0.0, 0.5]  # Point ahead of ROV towards obstacles
CAM_STEP_ANGLE = 3.0
CAM_STEP_DIST = 0.15
CAM_STEP_PAN = 0.08

# Auto-follow the ROV so it stays in view
CAM_FOLLOW = True
CAM_FOLLOW_Z_OFFSET = 0.15  # Increased for better view of ROV top

# ROV camera preview (rendered from a camera node in the GLTF if present)
CAM_PREVIEW_W = 320
CAM_PREVIEW_H = 240
CAM_FOV = 70.0
CAM_NEAR = 0.02
CAM_FAR = 15.0
PREVIEW_FPS = 12  # Reduced from 20 for less CPU load
VIS_FPS = 12      # Marker/overlay update rate (Hz)

# Backwards thrust scaling: fraction of forward power when running in reverse
BACKWARDS_THRUST_SCALE = 0.8

# Obstacles (movable props) — also treated as neutrally-buoyant objects in water
NUM_OBSTACLES = 6
OBSTACLE_SPAWN_CENTER = (1.2, 0.0, 0.55)  # Ahead of ROV start, in line of sight
OBSTACLE_SPREAD = (0.4, 0.5, 0.25)        # +/- spread around center
OBSTACLE_MASS = 2.0
OBSTACLE_SIZE_RANGE = (0.07, 0.14)        # box half-extent range (m)

# Make obstacles "float" in water (neutral buoyancy) and experience drag
OBSTACLE_BUOYANCY_SCALE = 1.00   # 1.0 = neutral (buoyancy equals weight)
OBSTACLE_DRAG_LIN = 6.0          # N per (m/s) (linear damping)
OBSTACLE_DRAG_QUAD = 10.0        # N per (m/s)^2 (quadratic damping)
OBSTACLE_MAX_DRAG = 120.0        # clamp for stability

# Terminal debug logging (prints state regularly)
LOG_FPS = 6                      # how often to print (Hz)
LOG_OBS = True                   # include obstacle info

# Detailed physics logging (structured, CSV-like lines written to the same log file)
LOG_PHYSICS_DETAILED = True     # write per-step structured physics lines (for optimization)
LOG_PHYSICS_HZ = 10             # how many detailed lines per second (approx)

# Marker spheres
MARKER_RADIUS = 0.045
MARKER_OFFSET = 0.00  # keep 0 to sit right on the CAD location
COLOR_OFF = [1.0, 0.15, 0.15, 0.0]  # OFF = invisible
COLOR_ON  = [0.15, 1.0, 0.15, 1.0]  # green

# Fallback thrusters if auto-detect fails (body frame: x forward, y left, z up)
THRUSTERS = [
    {"name":"thruster_1", "pos": ( 0.17,  0.11, -0.05), "dir": (1,0,0), "key": ord('1'), "kind":"H"},
    {"name":"thruster_2", "pos": (-0.17,  0.11,  0.00), "dir": (1,0,0), "key": ord('2'), "kind":"H"},
    {"name":"thruster_3", "pos": (-0.17, -0.11,  0.00), "dir": (1,0,0), "key": ord('3'), "kind":"H"},
    {"name":"thruster_4", "pos": ( 0.02,  0.00, -0.12), "dir": (0,0,1), "key": ord('4'), "kind":"V"},
]


# ==========================
# UTIL
# ==========================
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def deg2rad(d):
    return d * math.pi / 180.0

def vnorm(v):
    n = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) or 1.0
    return (v[0]/n, v[1]/n, v[2]/n)

def vcross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])

def mat4_mul(a, b):
    """Column-major 4x4 multiply: out = a*b (both len-16)."""
    out = [0.0] * 16
    # out[col,row] = sum_k a[k,row] * b[col,k]
    for col in range(4):
        for row in range(4):
            out[col*4 + row] = (
                a[0*4 + row] * b[col*4 + 0] +
                a[1*4 + row] * b[col*4 + 1] +
                a[2*4 + row] * b[col*4 + 2] +
                a[3*4 + row] * b[col*4 + 3]
            )
    return out

def quat_to_mat3(q):
    """(x,y,z,w) -> 3x3 (column-major)"""
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    # columns
    c0 = (1 - 2*(yy + zz), 2*(xy + wz),     2*(xz - wy))
    c1 = (2*(xy - wz),     1 - 2*(xx + zz), 2*(yz + wx))
    c2 = (2*(xz + wy),     2*(yz - wx),     1 - 2*(xx + yy))
    return c0, c1, c2

def trs_to_mat4(t, r, s):
    """glTF TRS to column-major 4x4."""
    tx, ty, tz = t
    sx, sy, sz = s
    c0, c1, c2 = quat_to_mat3(r)
    # apply scale
    c0 = (c0[0]*sx, c0[1]*sx, c0[2]*sx)
    c1 = (c1[0]*sy, c1[1]*sy, c1[2]*sy)
    c2 = (c2[0]*sz, c2[1]*sz, c2[2]*sz)
    return [
        c0[0], c0[1], c0[2], 0.0,
        c1[0], c1[1], c1[2], 0.0,
        c2[0], c2[1], c2[2], 0.0,
        tx,    ty,    tz,    1.0
    ]

def node_local_mat4(node):
    """Return column-major 4x4 for a node (matrix if present, else TRS, else identity)."""
    m = node.get("matrix", None)
    if isinstance(m, list) and len(m) == 16:
        return [float(x) for x in m]
    t = node.get("translation", [0.0, 0.0, 0.0])
    r = node.get("rotation",    [0.0, 0.0, 0.0, 1.0])
    s = node.get("scale",       [1.0, 1.0, 1.0])
    return trs_to_mat4(t, r, s)

# --- GLTF thruster transform helpers ---
def is_identity_trs(node):
    """True if node has no matrix and TRS is default."""
    if isinstance(node.get("matrix", None), list) and len(node["matrix"]) == 16:
        # if matrix exists, we consider it non-identity unless it is exactly identity
        m = node["matrix"]
        ident = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        return all(abs(float(m[i]) - ident[i]) < 1e-9 for i in range(16))
    t = node.get("translation", [0.0, 0.0, 0.0])
    r = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    s = node.get("scale", [1.0, 1.0, 1.0])
    return (abs(t[0]) < 1e-9 and abs(t[1]) < 1e-9 and abs(t[2]) < 1e-9 and
            abs(r[0]) < 1e-9 and abs(r[1]) < 1e-9 and abs(r[2]) < 1e-9 and abs(r[3] - 1.0) < 1e-9 and
            abs(s[0] - 1.0) < 1e-9 and abs(s[1] - 1.0) < 1e-9 and abs(s[2] - 1.0) < 1e-9)

def find_first_transform_descendant(nodes, root_idx):
    """
    Fusion often exports Thruster <k> as a group at origin, with the real occurrence transform
    on a descendant node (often named 'occurrence of ...') that has a non-identity matrix/TRS.
    Return the best node index to use for transform (root if nothing better found).
    """
    stack = [root_idx]
    best = root_idx
    while stack:
        i = stack.pop()
        n = nodes[i]
        nm = (n.get("name", "") or "").lower()
        # Prefer explicit occurrence transforms
        if ("occurrence" in nm or "thruster" in nm or "camera" in nm) and (not is_identity_trs(n)):
            return i
        if not is_identity_trs(n):
            best = i
        for c in (n.get("children", []) or []):
            stack.append(c)
    return best

def obj_bounds(path):
    vmin = [1e9, 1e9, 1e9]
    vmax = [-1e9, -1e9, -1e9]
    found = False
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    except ValueError:
                        continue
                    found = True
                    vmin[0] = min(vmin[0], x); vmin[1] = min(vmin[1], y); vmin[2] = min(vmin[2], z)
                    vmax[0] = max(vmax[0], x); vmax[1] = max(vmax[1], y); vmax[2] = max(vmax[2], z)
    if not found:
        center = (0.0, 0.0, 0.0)
        size = (0.4, 0.4, 0.25)
        return center, size
    center = ((vmin[0]+vmax[0])/2, (vmin[1]+vmax[1])/2, (vmin[2]+vmax[2])/2)
    size = (vmax[0]-vmin[0], vmax[1]-vmin[1], vmax[2]-vmin[2])
    return center, size

def gltf_mat_basis_and_pos(m):
    """
    glTF matrices are column-major 4x4 arrays length 16.
    Return basis columns (c0,c1,c2) and translation (t).
    """
    c0 = (m[0], m[1], m[2])
    c1 = (m[4], m[5], m[6])
    c2 = (m[8], m[9], m[10])
    t  = (m[12], m[13], m[14])
    return c0, c1, c2, t

def find_camera_pose_from_gltf(gltf_path, center):
    """
    Try to find a camera node/occurrence in the GLTF by name.
    Returns (pos_body, forward_body, up_body) in BODY frame, or None if not found.
    We look for node names containing 'camera' (case-insensitive).
    Uses accumulated node transforms (parents included).
    """
    if not os.path.exists(gltf_path):
        return None
    try:
        with open(gltf_path, "r", errors="ignore") as f:
            gltf = json.load(f)
    except Exception:
        return None

    nodes = gltf.get("nodes", []) or []
    if not nodes:
        return None

    # parent map
    parent = {}
    for i, n in enumerate(nodes):
        for c in (n.get("children", []) or []):
            parent[c] = i

    world_m = {}

    def world_mat(idx):
        if idx in world_m:
            return world_m[idx]
        local = node_local_mat4(nodes[idx])
        if idx in parent:
            wm = mat4_mul(world_mat(parent[idx]), local)
        else:
            wm = local
        world_m[idx] = wm
        return wm

    cam_candidates = []
    for i, n in enumerate(nodes):
        nm = (n.get("name", "") or "")
        if "camera" in nm.lower():
            cam_candidates.append(i)
    if not cam_candidates:
        return None

    idx = cam_candidates[0]
    m = world_mat(idx)

    c0, c1, c2, t = gltf_mat_basis_and_pos(m)

    forward = (-c2[0], -c2[1], -c2[2])
    up = c1

    pos_body = (t[0] - center[0], t[1] - center[1], t[2] - center[2])
    return (pos_body, vnorm(forward), vnorm(up))

def detect_thrusters_from_gltf(gltf_path, center):
    """
    Fusion-exported GLTF preserves named nodes:
      Thruster <1>, Thruster <2>, Thruster <3>, Thruster <4>
    Uses accumulated node transforms (parents included).
    We use:
      - translation as thruster position
      - Y-axis of matrix as thrust direction (matches thruster shaft axis)
    """
    if not os.path.exists(gltf_path):
        return []
    try:
        with open(gltf_path, "r", errors="ignore") as f:
            gltf = json.load(f)
    except Exception:
        return []

    nodes = gltf.get("nodes", []) or []
    if not nodes:
        return []

    # parent map
    parent = {}
    for i, n in enumerate(nodes):
        for c in (n.get("children", []) or []):
            parent[c] = i

    world_m = {}

    def world_mat(idx):
        if idx in world_m:
            return world_m[idx]
        local = node_local_mat4(nodes[idx])
        if idx in parent:
            wm = mat4_mul(world_mat(parent[idx]), local)
        else:
            wm = local
        world_m[idx] = wm
        return wm

    name_to_indices = {}
    for i, n in enumerate(nodes):
        nm = n.get("name", "")
        if nm:
            name_to_indices.setdefault(nm, []).append(i)

    thrusters = []
    for k in range(1, 5):
        nm = f"Thruster <{k}>"
        idx_list = name_to_indices.get(nm, [])
        if not idx_list:
            continue

        # Use the first candidate that yields a non-degenerate transform
        use_idx = None
        for cand in idx_list:
            cand2 = find_first_transform_descendant(nodes, cand)
            m_test = world_mat(cand2)
            _, _, _, t_test = gltf_mat_basis_and_pos(m_test)
            if (abs(t_test[0]) + abs(t_test[1]) + abs(t_test[2])) > 1e-6:
                use_idx = cand2
                m = m_test
                break
        if use_idx is None:
            cand2 = find_first_transform_descendant(nodes, idx_list[0])
            m = world_mat(cand2)

        c0, c1, c2, t = gltf_mat_basis_and_pos(m)

        px, py, pz = (t[0] - center[0], t[1] - center[1], t[2] - center[2])
        d = vnorm(c1)

        if abs(d[2]) > 0.70:
            kind = "V"
            if d[2] < 0:
                d = (-d[0], -d[1], -d[2])
        else:
            kind = "H"
            if d[0] < 0:
                d = (-d[0], -d[1], -d[2])

        thrusters.append({
            "name": f"thruster_{k}",
            "pos": (px, py, pz),
            "dir": d,
            "key": ord(str(k)),
            "kind": kind
        })

    if len(thrusters) != 4:
        return []
    thrusters.sort(key=lambda t: int(chr(t["key"])))
    return thrusters


# ==========================
# PHYSICS
# ==========================
def apply_buoyancy(body_id, base_pos, base_quat):
    """
    Apply buoyancy at the Center of Buoyancy (COB), offset above COM, to create righting torque.
    Buoyancy magnitude is scaled by BUOYANCY_SCALE (1.0 = neutral).
    """
    if not p.isConnected():
        return
    buoy = MASS * GRAVITY * BUOYANCY_SCALE

    # COB in world coordinates
    cob_rel_world = p.rotateVector(base_quat, COB_OFFSET_BODY)
    cob_world = (base_pos[0] + cob_rel_world[0],
                 base_pos[1] + cob_rel_world[1],
                 base_pos[2] + cob_rel_world[2])

    p.applyExternalForce(body_id, -1, (0.0, 0.0, buoy), cob_world, p.WORLD_FRAME)

# --- Ballast and righting ---
def apply_ballast(body_id, base_pos, base_quat):
    """Extra downward force applied below COM to emulate bottom-heavy ROV."""
    if not p.isConnected():
        return
    fz = -MASS * GRAVITY * BALLAST_SCALE
    rel_world = p.rotateVector(base_quat, BALLAST_OFFSET_BODY)
    p_world = (base_pos[0] + rel_world[0],
               base_pos[1] + rel_world[1],
               base_pos[2] + rel_world[2])
    p.applyExternalForce(body_id, -1, (0.0, 0.0, fz), p_world, p.WORLD_FRAME)

def apply_righting_torque(body_id, base_quat, ang_world):
    """
    Restore roll/pitch toward 0 using PD control (buoyancy-like righting).
    Prevents oscillation and excessive tilting.
    """
    if not p.isConnected():
        return

    roll, pitch, _ = p.getEulerFromQuaternion(base_quat)

    # Convert angular velocity to BODY frame for better damping control
    inv_q = p.invertTransform([0, 0, 0], base_quat)[1]
    w_body = p.rotateVector(inv_q, ang_world)
    p_rate = w_body[0]
    q_rate = w_body[1]

    # PD torque in BODY frame (proportional to error, derivative for damping)
    tx_b = -RIGHTING_K_RP * roll  - RIGHTING_KD_RP * p_rate
    ty_b = -RIGHTING_K_RP * pitch - RIGHTING_KD_RP * q_rate
    tz_b = 0.0

    # Clamp to avoid excessive torques
    tx_b = clamp(tx_b, -MAX_DRAG_TORQUE, MAX_DRAG_TORQUE)
    ty_b = clamp(ty_b, -MAX_DRAG_TORQUE, MAX_DRAG_TORQUE)

    # Convert BODY torque to WORLD and apply
    t_world = p.rotateVector(base_quat, (tx_b, ty_b, tz_b))
    p.applyExternalTorque(body_id, -1, t_world, p.WORLD_FRAME)

def apply_drag(body_id, base_pos, base_quat, lin_world, ang_world):
    """
    Improved drag model with linear + quadratic components.
    - Linear drag (low-speed damping)
    - Quadratic drag (high-speed hydrodynamic drag)
    - Rotational damping (WORLD + BODY frame)
    - All forces applied at COM to avoid spurious torques
    """
    if not p.isConnected():
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (False, False, False)

    # Relative flow velocity (account for water current)
    vrel_world = (
        lin_world[0] - WATER_CURRENT_WORLD[0],
        lin_world[1] - WATER_CURRENT_WORLD[1],
        lin_world[2] - WATER_CURRENT_WORLD[2],
    )

    # Convert to BODY frame for drag calculation
    inv_q = p.invertTransform([0, 0, 0], base_quat)[1]
    vrel_body = p.rotateVector(inv_q, vrel_world)

    # Estimate acceleration relative to water in BODY frame for added-mass effect.
    # Use a stored previous v_rel (LAST_VREL_BODY). The added-mass force is -A * a_rel.
    global LAST_VREL_BODY, LAST_A_BODY
    try:
        # Finite-difference accel (body-frame)
        ax_raw = (vrel_body[0] - LAST_VREL_BODY[0]) / DT
        ay_raw = (vrel_body[1] - LAST_VREL_BODY[1]) / DT
        az_raw = (vrel_body[2] - LAST_VREL_BODY[2]) / DT
    except Exception:
        ax_raw = ay_raw = az_raw = 0.0

    # Low-pass the acceleration estimate to avoid amplifying solver noise
    alpha = ACCEL_FILTER_ALPHA
    ax = alpha * ax_raw + (1.0 - alpha) * LAST_A_BODY[0]
    ay = alpha * ay_raw + (1.0 - alpha) * LAST_A_BODY[1]
    az = alpha * az_raw + (1.0 - alpha) * LAST_A_BODY[2]

    # Clamp to prevent explosion (but track saturation)
    vx_raw, vy_raw, vz_raw = vrel_body
    vx = clamp(vx_raw, -MAX_SPEED, MAX_SPEED)
    vy = clamp(vy_raw, -MAX_SPEED, MAX_SPEED)
    vz = clamp(vz_raw, -MAX_SPEED, MAX_SPEED)

    sat_speed = (abs(vx_raw) > MAX_SPEED or abs(vy_raw) > MAX_SPEED or abs(vz_raw) > MAX_SPEED)

    # Linear + quadratic drag in BODY frame
    fx_b = -LIN_DRAG_BODY[0]*vx - 0.5*RHO*CD[0]*AREA[0]*abs(vx)*vx
    fy_b = -LIN_DRAG_BODY[1]*vy - 0.5*RHO*CD[1]*AREA[1]*abs(vy)*vy
    fz_b = -LIN_DRAG_BODY[2]*vz - 0.5*RHO*CD[2]*AREA[2]*abs(vz)*vz

    # Added-mass forces in BODY frame (opposes acceleration of the vehicle relative to fluid)
    # F_added = -A * a_rel
    fa_x = -ADDED_MASS_BODY[0] * ax
    fa_y = -ADDED_MASS_BODY[1] * ay
    fa_z = -ADDED_MASS_BODY[2] * az

    fx_b += fa_x
    fy_b += fa_y
    fz_b += fa_z

    # Clamp drag forces for stability
    fx_b = clamp(fx_b, -MAX_DRAG_FORCE, MAX_DRAG_FORCE)
    fy_b = clamp(fy_b, -MAX_DRAG_FORCE, MAX_DRAG_FORCE)
    fz_b = clamp(fz_b, -MAX_DRAG_FORCE, MAX_DRAG_FORCE)

    # Convert drag force back to WORLD and apply at COM
    f_world = p.rotateVector(base_quat, (fx_b, fy_b, fz_b))
    p.applyExternalForce(body_id, -1, f_world, base_pos, p.WORLD_FRAME)

    # Also apply the added-mass component explicitly (already added above, but keep
    # an explicit world-frame application in case the tuning prefers seeing it separately)
    # Note: This is redundant with adding fa_* to fx_b, so we do not re-apply here to avoid doubling.

    # Rotational damping: WORLD-frame baseline
    wx = clamp(ang_world[0], -MAX_OMEGA, MAX_OMEGA)
    wy = clamp(ang_world[1], -MAX_OMEGA, MAX_OMEGA)
    wz = clamp(ang_world[2], -MAX_OMEGA, MAX_OMEGA)

    tx_w = -ROT_DAMP * wx
    ty_w = -ROT_DAMP * wy
    tz_w = -ROT_DAMP * wz

    # Body-frame angular damping (stronger for roll/pitch)
    w_body = p.rotateVector(inv_q, (wx, wy, wz))
    tx_b = -ANG_DAMP_BODY[0] * w_body[0]
    ty_b = -ANG_DAMP_BODY[1] * w_body[1]
    tz_b = -ANG_DAMP_BODY[2] * w_body[2]
    t_body_world = p.rotateVector(base_quat, (tx_b, ty_b, tz_b))

    # Combined torque with clamping
    tx = clamp(tx_w + t_body_world[0], -MAX_DRAG_TORQUE, MAX_DRAG_TORQUE)
    ty = clamp(ty_w + t_body_world[1], -MAX_DRAG_TORQUE, MAX_DRAG_TORQUE)
    tz = clamp(tz_w + t_body_world[2], -MAX_DRAG_TORQUE, MAX_DRAG_TORQUE)

    p.applyExternalTorque(body_id, -1, (tx, ty, tz), p.WORLD_FRAME)

    # Update stored previous relative velocity and acceleration for next-step finite difference
    try:
        LAST_VREL_BODY = (vrel_body[0], vrel_body[1], vrel_body[2])
        LAST_A_BODY = (ax, ay, az)
    except Exception:
        LAST_VREL_BODY = (0.0, 0.0, 0.0)
        LAST_A_BODY = (0.0, 0.0, 0.0)

    return f_world, (tx, ty, tz), (False, False, sat_speed)


# --- Obstacles: neutral buoyancy + drag in water ---
def apply_obstacle_water_forces(obstacle_ids):
    """Apply neutral buoyancy + drag to obstacles so they behave like objects in water."""
    if not p.isConnected() or not obstacle_ids:
        return
    for oid in obstacle_ids:
        try:
            pos, quat = p.getBasePositionAndOrientation(oid)
            lin, ang = p.getBaseVelocity(oid)
        except Exception:
            continue

        # Neutral buoyancy: upward force equals weight * scale
        buoy = OBSTACLE_MASS * GRAVITY * OBSTACLE_BUOYANCY_SCALE
        p.applyExternalForce(oid, -1, (0.0, 0.0, buoy), pos, p.WORLD_FRAME)

        # Drag opposing motion (WORLD frame)
        vx, vy, vz = lin
        v = math.sqrt(vx*vx + vy*vy + vz*vz)
        if v > 1e-6:
            # linear + quadratic
            mag = OBSTACLE_DRAG_LIN * v + OBSTACLE_DRAG_QUAD * v * v
            mag = clamp(mag, 0.0, OBSTACLE_MAX_DRAG)
            fx = -mag * (vx / v)
            fy = -mag * (vy / v)
            fz = -mag * (vz / v)
            p.applyExternalForce(oid, -1, (fx, fy, fz), pos, p.WORLD_FRAME)

        # Light angular damping (helps them settle)
        wx, wy, wz = ang
        tq = (-0.4 * wx, -0.4 * wy, -0.4 * wz)
        p.applyExternalTorque(oid, -1, tq, p.WORLD_FRAME)


# ==========================
# BUILD
# ==========================
def build_rov():
    if not os.path.exists(OBJ_FILE):
        raise FileNotFoundError(f"Missing mesh: {OBJ_FILE}")

    center, size = obj_bounds(OBJ_FILE)

    half = (
        max(0.01, abs(size[0]) * MESH_SCALE[0] / 2),
        max(0.01, abs(size[1]) * MESH_SCALE[1] / 2),
        max(0.01, abs(size[2]) * MESH_SCALE[2] / 2),
    )

    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)

    # Visual: keep mesh coordinates in body frame (centered by visualFramePosition)
    try:
        vis = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=OBJ_FILE,
            meshScale=MESH_SCALE,
            rgbaColor=[0.75, 0.78, 0.82, 1.0],
            visualFramePosition=(-center[0], -center[1], -center[2]),
            visualFrameOrientation=(0, 0, 0, 1),
        )
    except Exception:
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.75, 0.78, 0.82, 1.0])
        print("[WARN] Mesh visual failed to load. Using box visual fallback.")

    rov = p.createMultiBody(
        baseMass=MASS,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[0, 0, 0.60],
        baseOrientation=p.getQuaternionFromEuler([
            deg2rad(MESH_BODY_EULER_DEG[0]),
            deg2rad(MESH_BODY_EULER_DEG[1]),
            deg2rad(MESH_BODY_EULER_DEG[2]),
        ])
    )

    # Keep awake + no builtin damping (we do our own)
    p.changeDynamics(rov, -1, linearDamping=0.0, angularDamping=0.0)
    try:
        p.changeDynamics(rov, -1, activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)
    except Exception:
        pass

    print(f"Mesh bounds size (m): {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f}")
    print(f"Collision box half-extents (m): {half[0]:.3f}, {half[1]:.3f}, {half[2]:.3f}")
    print(f"Body rotation applied (deg r,p,y): {MESH_BODY_EULER_DEG}")
    return rov, center

def make_markers(thrusters):
    """Create marker spheres if rendering is enabled."""
    if not ENABLE_MARKERS:
        return []
    markers = []
    for _ in thrusters:
        sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=MARKER_RADIUS, rgbaColor=COLOR_OFF)
        m = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_vis, basePosition=[0, 0, 0])
        markers.append(m)
    return markers

# --- Obstacles ---
def spawn_obstacles(n=NUM_OBSTACLES):
    """Spawn simple dynamic boxes/spheres you can drag with the mouse or move via keys."""
    ids = []
    rng = random.Random(7)
    # create a few shapes
    for i in range(n):
        # random size
        s0 = rng.uniform(OBSTACLE_SIZE_RANGE[0], OBSTACLE_SIZE_RANGE[1])
        half = (s0, s0, s0)
        # random position near spawn center
        px = OBSTACLE_SPAWN_CENTER[0] + rng.uniform(-OBSTACLE_SPREAD[0], OBSTACLE_SPREAD[0])
        py = OBSTACLE_SPAWN_CENTER[1] + rng.uniform(-OBSTACLE_SPREAD[1], OBSTACLE_SPREAD[1])
        pz = OBSTACLE_SPAWN_CENTER[2] + rng.uniform(-OBSTACLE_SPREAD[2], OBSTACLE_SPREAD[2])
        # alternate box / sphere
        if i % 2 == 0:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.35, 0.55, 0.95, 1.0])
        else:
            r = s0
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[0.95, 0.55, 0.35, 1.0])
        bid = p.createMultiBody(baseMass=OBSTACLE_MASS, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                basePosition=[px, py, pz])
        p.changeDynamics(bid, -1, linearDamping=0.02, angularDamping=0.02, lateralFriction=0.7, restitution=0.05)
        ids.append(bid)
    return ids

def set_marker(marker_id, on):
    """Change marker color (only if markers are enabled)."""
    if not p.isConnected() or not ENABLE_MARKERS:
        return
    p.changeVisualShape(marker_id, -1, rgbaColor=(COLOR_ON if on else COLOR_OFF))

def update_marker_pose(marker_id, base_pos, base_quat, thr):
    """Update marker position at thruster location (only if markers are enabled)."""
    if not p.isConnected() or not ENABLE_MARKERS:
        return
    local = (
        thr["pos"][0] + thr["dir"][0] * MARKER_OFFSET,
        thr["pos"][1] + thr["dir"][1] * MARKER_OFFSET,
        thr["pos"][2] + thr["dir"][2] * MARKER_OFFSET,
    )
    rel = p.rotateVector(base_quat, local)
    world = (base_pos[0] + rel[0], base_pos[1] + rel[1], base_pos[2] + rel[2])
    p.resetBasePositionAndOrientation(marker_id, world, [0, 0, 0, 1])

def update_arrow(base_pos, base_quat, thr, level, arrow_id):
    """Update thruster force direction arrow (only if enabled)."""
    if not p.isConnected() or not ENABLE_THRUSTER_ARROWS:
        return arrow_id
    rel = p.rotateVector(base_quat, thr["pos"])
    p_world = (base_pos[0] + rel[0], base_pos[1] + rel[1], base_pos[2] + rel[2])
    d_world = p.rotateVector(base_quat, thr["dir"])
    tip = (p_world[0] + d_world[0] * 0.35, p_world[1] + d_world[1] * 0.35, p_world[2] + d_world[2] * 0.35)
    # level: -1..1 (negative = reverse). Draw direction and color based on sign and magnitude.
    mag = abs(level)
    if mag < 1e-3:
        # hide / draw tiny point when off
        return p.addUserDebugLine(p_world, p_world, [0.2, 0.2, 0.2], 1, lifeTime=0, replaceItemUniqueId=arrow_id)
    # invert direction if negative (reverse)
    if level < 0:
        tip = (p_world[0] - d_world[0] * 0.35, p_world[1] - d_world[1] * 0.35, p_world[2] - d_world[2] * 0.35)
        color = [0.2, 0.6, 1.0]  # blue-ish for reverse
    else:
        color = [1.0, 0.65, 0.1]  # orange for forward
    # line width scales with magnitude (clamped)
    width = max(1, min(8, int(2 + mag * 6)))
    return p.addUserDebugLine(p_world, tip, color, width, lifeTime=0, replaceItemUniqueId=arrow_id)


def main():
    global _log_file_handle
    
    # Initialize log file
    try:
        _log_file_handle = open(LOG_FILE, 'w')
        _log_file_handle.write(f"ROV Simulator Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        _log_file_handle.write("=" * 80 + "\n\n")
        # Write a small CSV header for structured physics logging (easy to import)
        if LOG_PHYSICS_DETAILED and _log_file_handle is not None:
            _log_file_handle.write("# DETAILED_PHYSICS_CSV\n")
            hdr = (
                "time,step,px,py,pz, vx,vy,vz, vbx,vby,vbz, wx,wy,wz, wbx,wby,wbz,"
                "Fthr_x,Fthr_y,Fthr_z, Tthr_x,Tthr_y,Tthr_z, Fdrag_x,Fdrag_y,Fdrag_z, Tdrag_x,Tdrag_y,Tdrag_z,"
                "buoy_z,ballast_z,thr_levels\n"
            )
            _log_file_handle.write(hdr)
            _log_file_handle.flush()
        _log_file_handle.flush()
        print(f"📝 Logging to: {LOG_FILE}")
    except Exception as e:
        print(f"⚠️  Could not open log file: {e}")
    
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()

    # GUI prefs
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        # Improve visuals: enable shadows and slightly increase GUI quality where available
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        except Exception:
            pass
    except Exception:
        pass

    p.setGravity(0, 0, -GRAVITY)
    p.setPhysicsEngineParameter(fixedTimeStep=DT, numSolverIterations=80, numSubSteps=1)

    # Plane far below (no "invisible box")
    p.loadURDF("plane.urdf", [0, 0, -80])

    rov, mesh_center = build_rov()

    obstacles = spawn_obstacles(NUM_OBSTACLES)
    obs_idx = 0
    if obstacles:
        print("Obstacles: click-drag them with the mouse, or use controls below.")
        print("Obstacle controls: TAB select next, WASD move, Q/E up/down, X to reset selected.")

    global THRUSTERS
    if AUTO_DETECT_THRUSTERS:
        auto_thr = detect_thrusters_from_gltf(GLTF_FILE, mesh_center)
        if auto_thr:
            THRUSTERS = auto_thr
            # Fix direction sign for specific thrusters (user verified 1 and 4 were backwards)
            for t in THRUSTERS:
                if t.get("name") in ("thruster_1", "thruster_4"):
                    dx, dy, dz = t["dir"]
                    t["dir"] = (-dx, -dy, -dz)
            print("[AUTO] Thrusters detected from GLTF:")
            for t in THRUSTERS:
                print("   ", t)
            p.addUserDebugText("[AUTO] Thrusters from GLTF (positions + angled dirs).", [0, 0, 1.2],
                               textColorRGB=[0, 1, 0], textSize=1.4, lifeTime=0)
        else:
            print("[AUTO] GLTF thruster detect failed; using fallback THRUSTERS.")
            p.addUserDebugText("[AUTO] GLTF thrusters NOT found (fallback).", [0, 0, 1.2],
                               textColorRGB=[1, 0.2, 0.2], textSize=1.4, lifeTime=0)

    # ROV camera pose (BODY frame) from GLTF if possible
    cam_info = find_camera_pose_from_gltf(GLTF_FILE, mesh_center)
    if cam_info is None:
        # Fallback: camera near front, looking forward (+X) with up +Z
        cam_pos_body = (0.18, 0.0, 0.05)
        cam_fwd_body = (1.0, 0.0, 0.0)
        cam_up_body = (0.0, 0.0, 1.0)
        print("[CAM] GLTF camera node not found; using fallback ROV camera pose.")
    else:
        cam_pos_body, cam_fwd_body, cam_up_body = cam_info
        print("[CAM] Using camera pose from GLTF.")

    markers = make_markers(THRUSTERS)
    # Create placeholder arrows; we'll update their endpoints frequently.
    arrows = [p.addUserDebugLine([0, 0, 0], [0, 0, 0], [1, 0.65, 0.1], 5, lifeTime=0) for _ in THRUSTERS]

    thr_on = [False] * len(THRUSTERS)
    thr_reverse = [False] * len(THRUSTERS)  # Track reverse mode for each thruster
    thr_cmd = [0.0] * len(THRUSTERS)   # -1 to 1 command (neg=reverse, pos=forward)
    thr_level = [0.0] * len(THRUSTERS) # actual ramped level -1..1
    for i, m in enumerate(markers):
        set_marker(m, False)

    # AUTOTEST: If ROV_AUTOTEST=1 in the environment, simulate key actions so
    # users without GUI focus can validate thruster toggle/reverse logic.
    AUTOTEST = os.environ.get("ROV_AUTOTEST", "0") == "1"
    if AUTOTEST:
        print("[AUTOTEST] Running simulated key actions (ROV_AUTOTEST=1)")
        # schedule: (sim_step to run at) -> action id
        # we'll perform a small scripted sequence to toggle thrusters and reverse
        autotest_schedule = {
            int(0.2 / DT): "t1_on",
            int(1.0 / DT): "t1_rev",
            int(2.0 / DT): "t1_off",
            int(2.5 / DT): "t2_on",
            int(3.0 / DT): "t2_rev",
            int(4.0 / DT): "done",
        }

    # Camera (local state, no globals)
    cam_dist = CAM_DIST
    cam_yaw = CAM_YAW
    cam_pitch = CAM_PITCH
    cam_target = list(CAM_TARGET)
    cam_follow = CAM_FOLLOW
    p.resetDebugVisualizerCamera(cameraDistance=cam_dist, cameraYaw=cam_yaw, cameraPitch=cam_pitch, cameraTargetPosition=cam_target)

    sim_step = 0
    log_every = max(1, int(round((1.0 / DT) / LOG_FPS)))
    log_phys_every = max(1, int(round((1.0 / DT) / LOG_PHYSICS_HZ)))
    vis_every = max(1, int(round((1.0 / DT) / VIS_FPS)))
    prev_every = max(1, int(round((1.0 / DT) / PREVIEW_FPS)))
    # Use the general visual update rate for thruster visuals to avoid
    # a separate THRUSTER_VIS_FPS constant (kept minimal for performance).
    thr_vis_every = vis_every
    proj = p.computeProjectionMatrixFOV(fov=CAM_FOV, aspect=float(CAM_PREVIEW_W)/float(CAM_PREVIEW_H), nearVal=CAM_NEAR, farVal=CAM_FAR)
    last_cam_warn = False

    print("\n" + "🌊 "*20)
    print("╔" + "═"*68 + "╗")
    print("║" + " ROV UNDERWATER SIMULATOR - PyBullet Physics Engine ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    print("\n" + "─"*72)
    print("📋 COMPLETE KEYBOARD CONTROLS")
    print("─"*72)
    print("\n🔧 THRUSTER CONTROL:")
    print("  [1-4]        Toggle Thruster 1-4 (ON/OFF)")
    print("  [Shift+1-4]  Toggle Reverse Mode (state remembered even when thruster is OFF)")
    print("  [Z/X/C/V]      Alternate keys to toggle Reverse Mode (useful if Shift isn't detected)")
    print("  [5-8]         Alternate numeric toggles for Thrusters 1-4 (ON/OFF)")
    print("               Thruster 1-3: Horizontal | Thruster 4: Vertical (Heave)")
    
    print("\n📷 CAMERA CONTROL:")
    print("  [J]/[L]      Yaw camera left/right")
    print("  [I]/[K]      Pitch camera up/down")
    print("  [U]/[O]      Zoom camera in/out")
    print("  [N]/[M]      Pan camera target up/down")
    print("  [F]          Toggle camera follow mode (ON/OFF)")
    
    print("\n🚧 OBSTACLE MANIPULATION:")
    print("  [TAB]        Select next obstacle (cycles through all)")
    print("  [W]/[A]/[S]/[D]  Move selected obstacle forward/left/back/right")
    print("  [Q]/[E]      Move selected obstacle up/down (depth)")
    print("  [X]          Reset selected obstacle to random position")
    print("  [Mouse]      Click-drag obstacles with left mouse button")
    
    print("\n⏮️  SIMULATION CONTROL:")
    print("  [R]          Reset ROV to start position (faces obstacles)")
    print("  [ESC]        Quit simulator")
    
    print("\n⚙️  PHYSICS ENGINE:")
    print(f"  Timestep: {DT*1000:.1f}ms ({1.0/DT:.0f} Hz)  │  Mass: {MASS} kg  │  Gravity: {GRAVITY} m/s²")
    print(f"  Limits: Speed {MAX_SPEED}m/s  │  Rotation {MAX_OMEGA}rad/s")
    print(f"  Thrust: Horizontal {MAX_THRUST_H}N  │  Vertical {MAX_THRUST_V}N")
    
    print("\n💧 HYDRODYNAMICS:")
    print(f"  Buoyancy: {BUOYANCY_SCALE:.2f}× weight @ {COB_OFFSET_BODY}  │  Ballast: {BALLAST_SCALE:.2f}× @ {BALLAST_OFFSET_BODY}")
    print(f"  Righting: Kp={RIGHTING_K_RP}  Kd={RIGHTING_KD_RP}")
    print(f"  Drag: ρ={RHO}kg/m³  │  Linear: {LIN_DRAG_BODY}  │  CD/Area: {CD}/{AREA}")
    
    print("\n🎨 RENDERING:")
    status_str = ""
    if ENABLE_CAMERA_PREVIEW:
        status_str += f"📷 Camera {PREVIEW_FPS}fps  "
    if ENABLE_THRUSTER_ARROWS:
        status_str += f"→ Arrows {VIS_FPS}fps  "
    if ENABLE_MARKERS:
        status_str += f"● Markers  "
    if not status_str:
        status_str = "Minimal (arrows/markers disabled)"
    print(f"  {status_str}")
    
    print("\n" + "🌊 "*20)
    print()

    while p.isConnected():
        keys = p.getKeyboardEvents()

        # Exit
        if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
            # Mark that the user intentionally requested quit so an outer
            # restart loop won't relaunch the simulator automatically.
            try:
                USER_QUIT = True
            except Exception:
                pass
            break

        # Reset
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            # Start position: facing obstacles ahead, level orientation
            start_pos = [0, 0, 0.60]
            start_rpy = [0, 0, 0]  # Yaw 0 = facing forward (+X)
            p.resetBasePositionAndOrientation(rov, start_pos, p.getQuaternionFromEuler(start_rpy))
            p.resetBaseVelocity(rov, [0, 0, 0], [0, 0, 0])
            print(f"[SIM] Reset ROV to {start_pos} facing forward (yaw={start_rpy[2]:.0f}°)")

        # Toggle camera follow
                # (obstacle reset logic removed during refactor - no-op here)

        # Camera controls
        cam_changed = False
        if ord('j') in keys and keys[ord('j')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            cam_yaw -= CAM_STEP_ANGLE; cam_changed = True
        if ord('l') in keys and keys[ord('l')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            cam_yaw += CAM_STEP_ANGLE; cam_changed = True
        if ord('i') in keys and keys[ord('i')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            cam_pitch += CAM_STEP_ANGLE; cam_changed = True
        if ord('k') in keys and keys[ord('k')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            cam_pitch -= CAM_STEP_ANGLE; cam_changed = True
        if ord('u') in keys and keys[ord('u')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            cam_dist = max(0.5, cam_dist - CAM_STEP_DIST); cam_changed = True
        if ord('o') in keys and keys[ord('o')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            cam_dist = min(6.0, cam_dist + CAM_STEP_DIST); cam_changed = True
        if (not cam_follow) and ord('n') in keys and keys[ord('n')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            cam_target[2] += CAM_STEP_PAN; cam_changed = True
        if (not cam_follow) and ord('m') in keys and keys[ord('m')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            cam_target[2] -= CAM_STEP_PAN; cam_changed = True

        if cam_changed:
            p.resetDebugVisualizerCamera(cameraDistance=cam_dist, cameraYaw=cam_yaw, cameraPitch=cam_pitch, cameraTargetPosition=cam_target)

        # Toggle thrusters (1-4) or toggle reverse mode (Shift+1-4)
        # Also allow direct reverse toggles with Z/X/C/V or numeric 5-8 if Shift+numbers doesn't register
        rev_key_map = [ord('z'), ord('x'), ord('c'), ord('v')]
        rev_num_map = [ord('5'), ord('6'), ord('7'), ord('8')]
        for i, t in enumerate(THRUSTERS):
            # Primary toggle (numbers): always toggle FORWARD on/off
            if t["key"] in keys and keys[t["key"]] & p.KEY_WAS_TRIGGERED:
                # Toggle on/off, but always treat primary numbers as FORWARD
                thr_on[i] = not thr_on[i]
                if thr_on[i]:
                    # Turning ON: force forward direction
                    thr_reverse[i] = False
                    thr_cmd[i] = 1.0
                    # Ensure thr_level sign is positive
                    if thr_level[i] < 0.0:
                        thr_level[i] = -thr_level[i]
                    if abs(thr_level[i]) < 1e-6:
                        thr_level[i] = 1e-3
                else:
                    # Immediately cut commanded and actual level for instant stop
                    thr_cmd[i] = 0.0
                    thr_level[i] = 0.0
                if i < len(markers):
                    set_marker(markers[i], thr_on[i])
                print(f'{t["name"]}: {"ON (FORWARD)" if thr_on[i] else "OFF"}')

            # Alternate reverse toggle keys (helpful if Shift isn't detected): z/x/c/v -> thrusters 1..4
            if i < len(rev_key_map) and rev_key_map[i] in keys and keys[rev_key_map[i]] & p.KEY_WAS_TRIGGERED:
                thr_reverse[i] = not thr_reverse[i]
                if thr_on[i]:
                    thr_cmd[i] = -1.0 if thr_reverse[i] else 1.0
                    thr_level[i] = -thr_level[i]
                print(f'{t["name"]}: {"REVERSE" if thr_reverse[i] else "FORWARD"} (via key)')

            # Numeric alternate keys 5-8: toggle reverse mode / reverse-on
            if i < len(rev_num_map) and rev_num_map[i] in keys and keys[rev_num_map[i]] & p.KEY_WAS_TRIGGERED:
                if not thr_on[i]:
                    # Turn ON in reverse
                    thr_reverse[i] = True
                    thr_on[i] = True
                    thr_cmd[i] = -1.0
                    # ensure thr_level is negative when turning on reverse
                    if thr_level[i] > 0.0:
                        thr_level[i] = -thr_level[i]
                    if abs(thr_level[i]) < 1e-6:
                        thr_level[i] = -1e-3
                    if i < len(markers):
                        set_marker(markers[i], True)
                    print(f'{t["name"]}: REVERSE (via key 5-8)')
                elif thr_on[i] and thr_reverse[i]:
                    # If already ON in reverse, turn OFF
                    thr_on[i] = False
                    thr_cmd[i] = 0.0
                    thr_level[i] = 0.0
                    if i < len(markers):
                        set_marker(markers[i], False)
                    print(f'{t["name"]}: OFF (via key 5-8)')
                else:
                    # Was ON forward -> switch to reverse immediately
                    thr_reverse[i] = True
                    thr_cmd[i] = -1.0
                    if thr_level[i] > 0.0:
                        thr_level[i] = -thr_level[i]
                    if abs(thr_level[i]) < 1e-6:
                        thr_level[i] = -1e-3
                    print(f'{t["name"]}: REVERSE (via key 5-8)')

        # Read state (guard against solver loss)
        try:
            base_pos, base_quat = p.getBasePositionAndOrientation(rov)
            lin, ang = p.getBaseVelocity(rov)
        except Exception:
            if not p.isConnected():
                print("\n[ERROR] PyBullet connection lost (possible physics server crash).")
                break
            print("\n[ERROR] Physics solver instability detected (getBaseVelocity failed).")
            print("[HINT] Try: lowering MAX_THRUST_H/MAX_THRUST_V, increasing ROT_DAMP, or reducing MAX_OMEGA.\n")
            break

        # Auto-follow: keep camera tracking the ROV with proper offset
        if cam_follow:
            # Camera leads the ROV slightly in the direction of motion (or stays ahead if stationary)
            follow_lead = 0.3  # Meters ahead
            cam_target[0] = base_pos[0] + follow_lead
            cam_target[1] = base_pos[1]
            cam_target[2] = base_pos[2] + CAM_FOLLOW_Z_OFFSET
            p.resetDebugVisualizerCamera(cameraDistance=cam_dist, cameraYaw=cam_yaw, cameraPitch=cam_pitch, cameraTargetPosition=cam_target)

        # ROV camera preview (separate window if OpenCV available)
        if ENABLE_CAMERA_PREVIEW and (sim_step % prev_every) == 0:
            # Camera pose in WORLD
            cam_pos_world = p.rotateVector(base_quat, cam_pos_body)
            cam_pos_world = (base_pos[0] + cam_pos_world[0], base_pos[1] + cam_pos_world[1], base_pos[2] + cam_pos_world[2])

            cam_fwd_world = p.rotateVector(base_quat, cam_fwd_body)
            cam_up_world = p.rotateVector(base_quat, cam_up_body)

            cam_target_pt = (cam_pos_world[0] + cam_fwd_world[0],
                             cam_pos_world[1] + cam_fwd_world[1],
                             cam_pos_world[2] + cam_fwd_world[2])

            view = p.computeViewMatrix(cam_pos_world, cam_target_pt, cam_up_world)
            img = p.getCameraImage(CAM_PREVIEW_W, CAM_PREVIEW_H, viewMatrix=view, projectionMatrix=proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            if img is None:
                continue
            rgba = img[2]  # width*height*4
            if rgba is None:
                continue
            if cv2 is not None:
                # Convert RGBA to BGR for OpenCV
                frame = rgba[:, :, :3][:, :, ::-1]
                cv2.imshow("ROV Camera", frame)
                cv2.waitKey(1)
            else:
                if not last_cam_warn:
                    print("[CAM] OpenCV not installed; cannot show live preview window. Install with: pip install opencv-python")
                    last_cam_warn = True

        # If autotest is enabled, perform scheduled simulated actions (no GUI keys needed)
        if 'AUTOTEST' in globals() and AUTOTEST and (sim_step in autotest_schedule):
            act = autotest_schedule[sim_step]
            if act == "t1_on":
                thr_on[0] = True
                thr_cmd[0] = -1.0 if thr_reverse[0] else 1.0
                if 0 < len(markers):
                    set_marker(markers[0], True)
                print("[AUTOTEST] thruster_1: ON")
            elif act == "t1_rev":
                thr_reverse[0] = not thr_reverse[0]
                if thr_on[0]:
                    thr_cmd[0] = -1.0 if thr_reverse[0] else 1.0
                    thr_level[0] = -thr_level[0]
                print(f"[AUTOTEST] thruster_1: {'REVERSE' if thr_reverse[0] else 'FORWARD'}")
            elif act == "t1_off":
                thr_on[0] = False
                thr_cmd[0] = 0.0
                thr_level[0] = 0.0
                if 0 < len(markers):
                    set_marker(markers[0], False)
                print("[AUTOTEST] thruster_1: OFF")
            elif act == "t2_on":
                thr_on[1] = True
                thr_cmd[1] = -1.0 if thr_reverse[1] else 1.0
                if 1 < len(markers):
                    set_marker(markers[1], True)
                print("[AUTOTEST] thruster_2: ON")
            elif act == "t2_rev":
                thr_reverse[1] = not thr_reverse[1]
                if thr_on[1]:
                    thr_cmd[1] = -1.0 if thr_reverse[1] else 1.0
                    thr_level[1] = -thr_level[1]
                print(f"[AUTOTEST] thruster_2: {'REVERSE' if thr_reverse[1] else 'FORWARD'}")
            elif act == "done":
                print("[AUTOTEST] Sequence complete — leaving simulator running (AUTOTEST disabled)")
                # Do not exit the simulator after the autotest sequence; disable AUTOTEST
                AUTOTEST = False

        # Update visuals. Markers updated at vis_every; thruster arrows updated at thr_vis_every
        if (sim_step % vis_every) == 0:
            for i, t in enumerate(THRUSTERS):
                if i < len(markers):
                    update_marker_pose(markers[i], base_pos, base_quat, t)
        if (sim_step % thr_vis_every) == 0:
            for i, t in enumerate(THRUSTERS):
                arrows[i] = update_arrow(base_pos, base_quat, t, thr_level[i], arrows[i])

        # Forces (hydrostatics + water drag)
        apply_buoyancy(rov, base_pos, base_quat)
        apply_ballast(rov, base_pos, base_quat)
        apply_righting_torque(rov, base_quat, ang)
        F_drag, T_drag, sat_drag = apply_drag(rov, base_pos, base_quat, lin, ang)

        # Compute force/torque budget terms for logging (WORLD frame)
        W = MASS * GRAVITY
        F_buoy = (0.0, 0.0, W * BUOYANCY_SCALE)
        F_ball = (0.0, 0.0, -W * BALLAST_SCALE)
        # Gravity acts at COM (no torque about COM)
        F_grav = (0.0, 0.0, -W)

        # Positions of buoyancy/ballast application in WORLD (relative to COM)
        cob_rel_w = p.rotateVector(base_quat, COB_OFFSET_BODY)
        bal_rel_w = p.rotateVector(base_quat, BALLAST_OFFSET_BODY)

        # Righting torque (WORLD) as applied in apply_righting_torque (PD)
        roll, pitch, yaw = p.getEulerFromQuaternion(base_quat)
        inv_q_log = p.invertTransform([0, 0, 0], base_quat)[1]
        w_body_log = p.rotateVector(inv_q_log, ang)
        tx_b = -RIGHTING_K_RP * roll  - RIGHTING_KD_RP * w_body_log[0]
        ty_b = -RIGHTING_K_RP * pitch - RIGHTING_KD_RP * w_body_log[1]
        sat_right = (abs(tx_b) > MAX_DRAG_TORQUE) or (abs(ty_b) > MAX_DRAG_TORQUE)
        tx_b = clamp(tx_b, -MAX_DRAG_TORQUE, MAX_DRAG_TORQUE)
        ty_b = clamp(ty_b, -MAX_DRAG_TORQUE, MAX_DRAG_TORQUE)
        T_right = p.rotateVector(base_quat, (tx_b, ty_b, 0.0))

        # Net thruster force/torque accumulators for debug
        # Compute body-frame velocity once for thruster inflow calculations
        inv_q_thr = p.invertTransform([0, 0, 0], base_quat)[1]
        v_body_for_thr = p.rotateVector(inv_q_thr, lin)

        F_thr_total = (0.0, 0.0, 0.0)
        T_thr_total = (0.0, 0.0, 0.0)

        for i, t in enumerate(THRUSTERS):
            # First-order ramp to command with asymmetric time constants
            # Now supports -1 (reverse) to +1 (forward)
            if thr_cmd[i] > thr_level[i]:
                tau = THRUSTER_TAU_UP
            elif thr_cmd[i] < thr_level[i]:
                tau = THRUSTER_TAU_DN
            else:
                tau = THRUSTER_TAU_DN  # default
            thr_level[i] += (DT / max(1e-6, tau)) * (thr_cmd[i] - thr_level[i])
            thr_level[i] = clamp(thr_level[i], -1.0, 1.0)  # Allow negative (reverse)
            if abs(thr_level[i]) <= 1e-4:
                continue

            thrust_max = MAX_THRUST_H if t["kind"] == "H" else MAX_THRUST_V
            thrust = thrust_max * thr_level[i]  # Negative thrust = reverse
            # Scale reverse thrust magnitude to be a bit lower than forward
            if thrust < 0.0:
                thrust *= BACKWARDS_THRUST_SCALE

            # Simple empirical thrust loss due to inflow: reduce thrust as the
            # local inflow speed along the propeller axis increases.
            # Use body-frame velocity and thruster direction (which is stored in body frame).
            dir_body = t.get("dir", (1.0, 0.0, 0.0))
            # speed along prop axis (positive = flow in same direction)
            speed_along = v_body_for_thr[0]*dir_body[0] + v_body_for_thr[1]*dir_body[1] + v_body_for_thr[2]*dir_body[2]
            loss = THRUSTER_SPEED_LOSS_COEF * abs(speed_along)
            loss = clamp(loss, 0.0, 0.9)  # don't zero-out thrust fully
            thrust *= (1.0 - loss)

            dir_world = p.rotateVector(base_quat, t["dir"])
            force = (dir_world[0] * thrust, dir_world[1] * thrust, dir_world[2] * thrust)

            # Accumulate net thruster force/torque (about COM) for debug
            F_thr_total = (F_thr_total[0] + force[0], F_thr_total[1] + force[1], F_thr_total[2] + force[2])
            rel_pos_world = p.rotateVector(base_quat, t["pos"])
            tau = vcross(rel_pos_world, force)
            T_thr_total = (T_thr_total[0] + tau[0], T_thr_total[1] + tau[1], T_thr_total[2] + tau[2])
            world_pos = (base_pos[0] + rel_pos_world[0],
                         base_pos[1] + rel_pos_world[1],
                         base_pos[2] + rel_pos_world[2])

            p.applyExternalForce(rov, -1, force, world_pos, p.WORLD_FRAME)

        # Structured, per-step physics logging (CSV-like) to help tuning/optimization
        if LOG_PHYSICS_DETAILED and (sim_step % log_phys_every) == 0 and _log_file_handle is not None:
            try:
                sim_t = sim_step * DT
                inv_q_csv = p.invertTransform([0, 0, 0], base_quat)[1]
                v_body_csv = p.rotateVector(inv_q_csv, lin)
                w_body_csv = p.rotateVector(inv_q_csv, ang)
                thr_levels_str = ";".join([f"{x:.3f}" for x in thr_level])
                line = (
                    f"{sim_t:.4f},{sim_step},"
                    f"{base_pos[0]:.4f},{base_pos[1]:.4f},{base_pos[2]:.4f},"
                    f"{lin[0]:.4f},{lin[1]:.4f},{lin[2]:.4f},"
                    f"{v_body_csv[0]:.4f},{v_body_csv[1]:.4f},{v_body_csv[2]:.4f},"
                    f"{ang[0]:.4f},{ang[1]:.4f},{ang[2]:.4f},"
                    f"{w_body_csv[0]:.4f},{w_body_csv[1]:.4f},{w_body_csv[2]:.4f},"
                    f"{F_thr_total[0]:.3f},{F_thr_total[1]:.3f},{F_thr_total[2]:.3f},"
                    f"{T_thr_total[0]:.3f},{T_thr_total[1]:.3f},{T_thr_total[2]:.3f},"
                    f"{F_drag[0]:.3f},{F_drag[1]:.3f},{F_drag[2]:.3f},"
                    f"{T_drag[0]:.3f},{T_drag[1]:.3f},{T_drag[2]:.3f},"
                    f"{F_buoy[2]:.3f},{F_ball[2]:.3f},\"{thr_levels_str}\"\n"
                )
                _log_file_handle.write(line)
                _log_file_handle.flush()
            except Exception:
                # Keep logging best-effort; never crash sim for logging errors
                pass

        # Periodic debug logging so we can see what is happening from the terminal output
        if (sim_step % log_every) == 0:
            roll, pitch, yaw = p.getEulerFromQuaternion(base_quat)
            spd = math.sqrt(lin[0]*lin[0] + lin[1]*lin[1] + lin[2]*lin[2])
            omg = math.sqrt(ang[0]*ang[0] + ang[1]*ang[1] + ang[2]*ang[2])
            
            # Formatted telemetry output
            print(f"\n┌─ t={sim_step*DT:7.2f}s ─────────────────────────────────────────────────┐")
            print(f"│ 📍 Position: ({base_pos[0]:+6.2f}, {base_pos[1]:+6.2f}, {base_pos[2]:+6.2f}) m")
            print(f"│ 🚀 Velocity: ({lin[0]:+6.2f}, {lin[1]:+6.2f}, {lin[2]:+6.2f}) m/s  │v│={spd:5.2f}")
            print(f"│ 🔄 Attitude: RPY=({math.degrees(roll):+7.1f}°, {math.degrees(pitch):+7.1f}°, {math.degrees(yaw):+7.1f}°)")
            print(f"│ ⚙️  Angular: ω=({ang[0]:+6.2f}, {ang[1]:+6.2f}, {ang[2]:+6.2f}) rad/s  │ω│={omg:5.2f}")
            
            # Thrusters on one line
            thr_status = " ".join([
                f"T{i+1}:{thr_level[i]:4.2f}" if thr_on[i] else f"T{i+1}:----"
                for i in range(len(THRUSTERS))
            ])
            print(f"│ 🔧 Thrusters: {thr_status}")
            
            # Force/torque budget (WORLD)
            F_net = (F_grav[0] + F_buoy[0] + F_ball[0] + F_drag[0] + F_thr_total[0],
                     F_grav[1] + F_buoy[1] + F_ball[1] + F_drag[1] + F_thr_total[1],
                     F_grav[2] + F_buoy[2] + F_ball[2] + F_drag[2] + F_thr_total[2])

            # Torques about COM from buoyancy/ballast + righting + drag + thrusters
            T_buoy = vcross(cob_rel_w, F_buoy)
            T_ball = vcross(bal_rel_w, F_ball)
            T_net = (T_buoy[0] + T_ball[0] + T_right[0] + T_drag[0] + T_thr_total[0],
                     T_buoy[1] + T_ball[1] + T_right[1] + T_drag[1] + T_thr_total[1],
                     T_buoy[2] + T_ball[2] + T_right[2] + T_drag[2] + T_thr_total[2])

            # Condensed forces & torques on two lines
            print(f"│ 💪 Forces (N):  thr={F_thr_total[0]:+6.1f},{F_thr_total[1]:+6.1f},{F_thr_total[2]:+6.1f}  "
                  f"drag={F_drag[0]:+6.1f},{F_drag[1]:+6.1f},{F_drag[2]:+6.1f}  "
                  f"net={F_net[0]:+6.1f},{F_net[1]:+6.1f},{F_net[2]:+6.1f}")
            print(f"│ 🌀 Torques (Nm): thr={T_thr_total[0]:+6.1f},{T_thr_total[1]:+6.1f},{T_thr_total[2]:+6.1f}  "
                  f"right={T_right[0]:+6.1f},{T_right[1]:+6.1f},{T_right[2]:+6.1f}")

            # Warnings
            warning_flags = []
            if spd > 0.85 * MAX_SPEED:
                warning_flags.append(f"⚠️  SPEED {spd:.2f}/{MAX_SPEED}m/s")
            if omg > 0.85 * MAX_OMEGA:
                warning_flags.append(f"⚠️  OMEGA {omg:.2f}/{MAX_OMEGA}rad/s")
            if sat_drag[2]:
                warning_flags.append("⚠️  DRAG_SAT")
            if sat_right:
                warning_flags.append("⚠️  RIGHT_SAT")
            
            if warning_flags:
                print(f"│ {' | '.join(warning_flags)}")

            if LOG_OBS and obstacles:
                try:
                    sel_id = obstacles[obs_idx]
                    o_pos, _ = p.getBasePositionAndOrientation(sel_id)
                    o_lin, _ = p.getBaseVelocity(sel_id)
                    print(f"│ 🎯 Obs[{obs_idx+1}/{len(obstacles)}]: pos=({o_pos[0]:+.2f},{o_pos[1]:+.2f},{o_pos[2]:+.2f}) "
                          f"v=({o_lin[0]:+.2f},{o_lin[1]:+.2f},{o_lin[2]:+.2f})")
                except Exception:
                    pass
            
            print(f"└────────────────────────────────────────────────────────────────────┘")

        # Water forces for obstacles (neutral buoyancy + drag)
        apply_obstacle_water_forces(obstacles)

        sim_step += 1
        if not p.isConnected():
            break
        p.stepSimulation()
        if SLEEP_REALTIME:
            time.sleep(DT)

    try:
        p.disconnect()
    except Exception:
        pass
    
    # Close log file
    if _log_file_handle is not None:
        try:
            _log_file_handle.write("\n" + "=" * 80 + "\n")
            _log_file_handle.write(f"Simulation ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            _log_file_handle.close()
            print(f"✅ Log saved to: {LOG_FILE}")
        except Exception:
            pass


if __name__ == "__main__":
    import traceback as _tb
    import time as _time
    # Run main() in a restart loop so transient crashes or accidental exits
    # don't immediately close the simulator. The user can press ESC to set
    # USER_QUIT=True and request an intentional exit.
    while True:
        try:
            main()
        except Exception as _e:
            print(f"[ERROR] Simulator crashed: {_e}")
            _tb.print_exc()
            print("Restarting simulator in 1s...")
            _time.sleep(1)
            continue

        # main() returned normally. If the user requested quit (ESC), honor it.
        try:
            if USER_QUIT:
                print("User requested exit (ESC). Quitting simulator.")
                break
        except Exception:
            # If anything goes wrong reading the flag, break to avoid infinite loops
            break

        # If we get here, main() exited without USER_QUIT set: treat as unexpected
        print("Simulator exited unexpectedly — restarting in 1s...")
        _time.sleep(1)

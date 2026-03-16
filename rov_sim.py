import os
import sys
import time
import math
import json
import random
import struct
import multiprocessing
from datetime import datetime

try:
    import cv2  # optional (used for camera preview window)
except ImportError:
    cv2 = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

import pybullet as p
import pybullet_data

from sim_shared import (
    CMD_RESET_ROV,
    CONTROL_MODE,
    CONTROL_MODE_BINARY,
    CONTROL_MODE_PROPORTIONAL,
    CTRL_H,
    CTRL_W,
    DEPTH_HOLD_ACTIVE,
    DEPTH_M,
    HEADING_DEG,
    HEADING_HOLD_ACTIVE,
    PITCH_RAD,
    REC_FLAG,
    REC_STATUS,
    REC_STATUS_FRAME_WRITE_FAILED,
    REC_STATUS_MISSING_DEPS,
    REC_STATUS_OK,
    REC_STATUS_WRITER_OPEN_FAILED,
    ROLL_RAD,
    SET_CAM_CHASE,
    SET_CAM_FOLLOW,
    SET_DEPTH_HOLD,
    SET_EMERGENCY_SURFACE,
    SET_HEADING_HOLD,
    SET_PROPORTIONAL_MODE,
    SET_SHOW_FORCE_VECTORS,
    SET_THRUST_LEVEL,
    SET_THRUSTER_FAILURE,
    SET_TOPDOWN,
    SET_TRAIL_ENABLED,
    SPEED_MPS,
    THRUST_LEVEL as SHM_THRUST_LEVEL,
)

# Joystick control panel (separate Tkinter window)
try:
    import joystick_panel
    HAS_JOYSTICK = True
except ImportError:
    HAS_JOYSTICK = False

# If the simulator is asked to quit via ESC, set this flag so an external
# restart loop does not immediately relaunch the simulator.
USER_QUIT = False

# ==========================
# FILE LOGGING SETUP
# ==========================
# Support PyInstaller frozen executables: use _MEIPASS if available
if getattr(sys, 'frozen', False):
    HERE = os.path.dirname(sys.executable)
    # Data files are unpacked to sys._MEIPASS by PyInstaller
    DATA_DIR = sys._MEIPASS
else:
    HERE = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = HERE
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
            # Don't flush every line — let the OS buffer handle it
    except (OSError, ValueError):
        # ValueError: I/O operation on closed file
        pass

"""
Underwater ROV sim (PyBullet) — GLTF-aligned thrusters

- Loads your CAD visual mesh (Assembly 1.obj) and uses a stable box collision.
- Neutral buoyancy (buoyancy force = weight).
- Quadratic drag + rotational damping (simple underwater model).
- 4 thrusters toggled by keys 1-4 (ON/OFF).
- Thruster markers: RED=off, GREEN=on, placed at thruster positions and aligned to thruster direction.
- Thruster placement + direction are taken from Assembly 1.gltf if present (nodes: Thruster <1..4>).

Camera:
- Mouse works normally.
- Keyboard: J/L yaw, I/K pitch, U/O zoom, N/M move target up/down.
- R resets, ESC quits.

Notes:
- Ensure Assembly 1.obj and Assembly 1.gltf are in the same folder as this script.
"""

# ==========================
# CONFIG
# ==========================
DT = 1/120  # 120 Hz physics timestep (smoother, more stable for light ROV)
SLEEP_REALTIME = True

GRAVITY = 9.81
MASS = 7.5  # kg — DDR spec: < 8 kg

# ============================================
# HYDROSTATICS: Buoyancy + Ballast for Self-Righting
# ============================================
COB_OFFSET_BODY = (0.0, 0.0, 0.08)   # meters above COM — DDR: CB above CG
BUOYANCY_SCALE  = 1.04               # slightly positive buoyancy per DDR spec

BALLAST_OFFSET_BODY = (0.0, 0.0, -0.10)  # meters below COM (battery on bottom per DDR)
BALLAST_SCALE       = 0.04               # realistic bottom-heavy bias (heavier battery pack)

# PD controller for roll/pitch righting (avoids oscillation)
# Realistic: buoyancy-ballast metacentric height creates a restoring torque
# of ~MASS*g*GM ≈ 7.5*9.81*0.03 ≈ 2.2 N·m/rad. We add a bit more for the
# ballast + CoB offset but keep it physically plausible — NOT a stiff spring.
RIGHTING_K_RP = 3.0    # N*m per rad — realistic metacentric restoring moment
RIGHTING_KD_RP = 2.5   # N*m per (rad/s) — gentle damping (water does most of the work)

# Water current BASE in WORLD frame (m/s). Set to (0,0,0) to disable.
WATER_CURRENT_BASE = (0.01, 0.005, 0.0)   # gentle background current — pool setting
# Time-varying ocean current parameters (sinusoidal variation)
CURRENT_VARIATION_AMP = (0.005, 0.003, 0.002)  # small variation in pool
CURRENT_VARIATION_PERIOD = (15.0, 20.0, 25.0)
# Runtime current (updated each step)
WATER_CURRENT_WORLD = list(WATER_CURRENT_BASE)

# ============================================
# THRUST CONFIGURATION — from Ctrl+Sea DDR performance predictions
# ============================================
MAX_THRUST_H = 5.56   # Horizontal thruster max force (N) per DDR
MAX_THRUST_V = 5.56   # Vertical thruster max force (N) per DDR

# Variable thrust level: user can adjust via +/- keys (scales MAX_THRUST)
THRUST_LEVEL = 1.0        # 0.0 to 1.0 multiplier
THRUST_LEVEL_STEP = 0.1   # step per keypress

# Asymmetric thruster ramping (realistic brushless DC motor spool-up in water)
# Real thrusters: ~0.7-1.5s to reach full RPM from stop, faster to spin down.
THRUSTER_TAU_UP = 0.70  # Ramp-up time constant (realistic motor spool-up)
THRUSTER_TAU_DN = 0.25  # Ramp-down time constant (motor braking + drag)

# ============================================
# FOSSEN HYDRODYNAMIC MODEL
# ============================================
# Based on Fossen, "Handbook of Marine Craft Hydrodynamics and Motion Control"
# (2011), and the UUV Simulator (uuvsimulator) implementation.
#
# The standard 6DOF marine vehicle equation of motion is:
#
#   M · ν̇  +  C(ν) · ν  +  D(ν) · ν  +  g(η)  =  τ
#
#   M    = M_RB + M_A        (rigid-body + added-mass inertia, 6×6)
#   C(ν) = C_RB(ν) + C_A(ν) (rigid-body + added-mass Coriolis/centripetal)
#   D(ν) = D_linear + D_nonlinear  (linear + quadratic hydrodynamic damping)
#   g(η) = restoring forces  (buoyancy − gravity, righting moments)
#   τ    = thruster forces & torques
#
# We use diagonal approximations for M_A and D since our box-like ROV has
# principal axes aligned with the body frame (low off-diagonal coupling).
# The Coriolis matrix C_A is NOT diagonal even for diagonal M_A — it
# captures the cross-coupling between translational and rotational motion
# through the added mass, and is the primary improvement over the old model.
# ============================================
RHO = 998.0  # Water density (kg/m³) per DDR

# --- D(ν): Damping ---
# Quadratic drag coefficients — from DDR: CD_frontal=1.5, CD_vertical=1.2
# Areas from DDR: frontal 0.0929 m², top/bottom 0.1238 m²
CD = [1.5, 1.5, 1.2]                  # surge, sway, heave
AREA = [0.0929, 0.0929, 0.1238]       # m² — measured from CAD per DDR

# Linear damping in BODY frame: D_linear · ν (N per m/s)
# Models viscous skin-friction at low Reynolds numbers + parasitic drag
# from frame struts, housings, etc. Ensures the ROV stops at low speeds
# where quadratic drag alone is negligible.
# Fossen (Sec 8.4.2): D_linear = diag(X_u, Y_v, Z_w, K_p, M_q, N_r)
LIN_DRAG_BODY = (4.8, 5.0, 6.0)       # translational (N per m/s)
LIN_DRAG_ANG  = (2.5, 2.5, 2.0)       # rotational   (N·m per rad/s)

# Quadratic (nonlinear) damping: D_nonlin · |ν| · ν
# For translational: F = ½ρ·Cd·A·|v|·v  (applied per axis in body frame)
# For rotational: T = Cd_rot · |ω| · ω  (significant at higher angular rates)
# Fossen (Sec 8.4.2): D_nonlinear = diag(X_u|u|, Y_v|v|, ...)
QUAD_DRAG_ANG = (3.5, 3.5, 2.5)       # rotational quadratic (N·m per (rad/s)²)

# --- M_A: Added mass (diagonal approximation) ---
# A 7.5 kg ROV displaces ~7.5 liters. For bluff underwater bodies, added mass
# is typically 60-100% of displaced-water mass per axis (higher for flat
# plates/sway, lower for streamlined surge). Fossen Sec 8.3.
# Effective translational mass = MASS + M_A, so surge effective = 7.5+7.5 = 15 kg.
ADDED_MASS_BODY = (7.5, 9.0, 10.0)    # X_udot, Y_vdot, Z_wdot (kg)

# Added rotational inertia (kg·m²): I_A = diag(K_pdot, M_qdot, N_rdot)
# Water entrained by rotation. For a box-like ROV ~0.35×0.49×0.34m,
# the added rotational inertia is significant.
ADDED_INERTIA_BODY = (0.08, 0.08, 0.06)  # roll, pitch, yaw

# --- C_A(ν): Added-mass Coriolis/centripetal ---
# Even with diagonal M_A, the Coriolis matrix is NOT diagonal.
# For a diagonal added mass M_A = diag(Xu, Yv, Zw, Kp, Mq, Nr),
# Fossen (Eq. 8.22) gives the skew-symmetric C_A matrix:
#
#   C_A(ν) = [ 0₃  -S(M_A11 · v1)  ]
#            [ -S(M_A11 · v1)  -S(M_A22 · v2) ]
#
# where v1=[u,v,w] (linear vel), v2=[p,q,r] (angular vel),
# M_A11=diag(Xu,Yv,Zw), M_A22=diag(Kp,Mq,Nr), and S(·) is the
# skew-symmetric (cross-product) matrix operator.
#
# This produces forces/torques:
#   F_coriolis_lin = -S(M_A11·v1)·ω - S(M_A22·v2)·ω  (forces from rotation×momentum)
#   T_coriolis     = -S(M_A11·v1)·v  (torques from velocity×momentum)
#
# These couple surge/sway/heave with roll/pitch/yaw through the water's
# added inertia — crucial for realistic turning and maneuvering behavior.
CORIOLIS_SCALE = 1.0    # 1.0 = full Fossen C_A, reduce to soften coupling

# Simple thrust loss coefficient from inflow speed along propeller axis
THRUSTER_SPEED_LOSS_COEF = 0.05   # realistic: thrusters lose ~5% per m/s of inflow

# Added-mass accel filter (Fossen/UUV Sim use filteredAcc with alpha to
# prevent instability from finite-difference acceleration estimates).
# Alpha=0.5 means the force tracks real acceleration with ~2-frame delay
# at 120 Hz, which is responsive enough for the inertia effect.
ACCEL_FILTER_ALPHA = 0.5

# Internal state: remember previous body-frame velocity relative to water
# for finite-difference acceleration estimates (added-mass force).
LAST_VREL_BODY = None  # Will be set on first sim step
LAST_A_BODY = (0.0, 0.0, 0.0)
LAST_W_BODY = None
LAST_ALPHA_BODY = (0.0, 0.0, 0.0)

# ============================================
# TETHER SIMULATION (DISABLED — removed to simplify physics)
# ============================================
TETHER_ENABLED = False

# ============================================
# DEPTH-DEPENDENT EFFECTS
# ============================================
DEPTH_BUOYANCY_COMPRESSIBILITY = 0.001   # fraction buoyancy loss per meter depth
SURFACE_Z = 1.2              # water surface z-coordinate (world) — pool depth ~3m
SEABED_Z = -2.0              # seabed plane z-coordinate — shallow pool/test tank

# ============================================
# WATER FOG / VISIBILITY
# ============================================
WATER_FOG_ENABLED = True
WATER_FOG_COLOR_SURFACE = (0.15, 0.45, 0.55)  # blue-green near surface
WATER_FOG_COLOR_DEEP = (0.04, 0.08, 0.12)     # dark at depth
WATER_FOG_DEPTH_RANGE = 20.0  # meters over which fog darkens fully

# ============================================
# HUD (ON-SCREEN TEXT OVERLAY)
# ============================================
HUD_ENABLED = False
HUD_UPDATE_HZ = 10  # Hz — now uses a single combined text item (1 IPC call)

# ============================================
# ON-BOARD CAMERA OVERLAY (depth/heading HUD burned into video feed)
# ============================================
CAMERA_OSD_ENABLED = True    # draw depth + heading onto the camera frame

# ============================================
# DEPTH-DEPENDENT CAMERA TINT (blue-green at depth)
# ============================================
CAMERA_DEPTH_TINT = True          # tint the onboard camera feed based on depth
CAMERA_TINT_MAX_ALPHA = 0.25      # max tint opacity at maximum depth
CAMERA_TINT_COLOR = (0, 40, 50)   # subtle dark blue-green (R, G, B — blended into frame)

# ============================================
# PROXIMITY WARNINGS (seabed / surface / obstacles)
# ============================================
PROXIMITY_WARN_DIST = 0.25    # meters — warn when closer than this to seabed/surface
PROXIMITY_WARN_ENABLED = True

# ============================================
# COLLISION DETECTION FEEDBACK
# ============================================
COLLISION_FEEDBACK_ENABLED = True  # print info when ROV contacts environment
COLLISION_LOG_COOLDOWN = 1.0       # seconds — suppress duplicate collision messages

# ============================================
# CAMERA SHAKE (vibration at speed)
# ============================================
CAMERA_SHAKE_ENABLED = True
CAMERA_SHAKE_INTENSITY = 0.004  # meters of random offset per 1 m/s speed

# ============================================
# EMERGENCY SURFACE (single-key ballistic ascent)
# ============================================
EMERGENCY_SURFACE_KEY = ord('0')   # press '0' for emergency surface

# ============================================
# HEADING-ALIGNED CHASE CAMERA
# ============================================
CAM_CHASE_ENABLED = False   # auto-rotate orbit camera to follow ROV heading
CAM_CHASE_SMOOTH = 0.03     # interpolation factor per frame (lower = smoother)

# ============================================
# THRUSTER EFFICIENCY TRACKING
# ============================================
# Tracks actual force output vs max possible (considering ramp + inflow loss)
# Displayed in telemetry log as percentage

# ============================================
# STABILITY CLAMPS — DDR max speed ~0.4 m/s
# ============================================
MAX_SPEED = 1.5    # m/s — generous headroom above DDR's ~0.4 m/s prediction
MAX_OMEGA = 4.0    # rad/s
MAX_DRAG_FORCE = 100.0   # Clamp on linear drag forces (N) — scaled for lighter ROV
MAX_DRAG_TORQUE = 50.0   # Clamp on damping torques (N*m)

# ============================================
# THRUSTER CONFIGURATIONS
# ============================================
# Each configuration maps a user-friendly name to its CAD model files.
# The GLTF drives thruster/camera auto-detection; the OBJ is the visual mesh.
THRUSTER_CONFIGS = {}

# Discover available configurations from CAD files present on disk.
# Versioned naming convention: v<N>.obj + v<N>.gltf (e.g., v1, v2, v3).
_versioned_cfgs = []
for _vn in os.listdir(DATA_DIR):
    _base, _ext = os.path.splitext(_vn)
    if _ext.lower() == ".obj" and os.path.exists(os.path.join(DATA_DIR, _base + ".gltf")):
        # Skip legacy "Assembly 1" if v-configs exist (will be added as fallback below)
        if _base.lower().startswith("v") and _base[1:].isdigit():
            _versioned_cfgs.append((int(_base[1:]), _base))

for _, _base in sorted(_versioned_cfgs, key=lambda x: x[0]):
    _label = f"Configuration {_base.upper()}"
    THRUSTER_CONFIGS[_label] = {
        "obj":  os.path.join(DATA_DIR, _base + ".obj"),
        "gltf": os.path.join(DATA_DIR, _base + ".gltf"),
    }

# Legacy "Assembly 1" as fallback if present
_legacy_obj = os.path.join(DATA_DIR, "Assembly 1.obj")
_legacy_gltf = os.path.join(DATA_DIR, "Assembly 1.gltf")
if os.path.exists(_legacy_obj) and os.path.exists(_legacy_gltf):
    THRUSTER_CONFIGS["Assembly 1 (Legacy)"] = {
        "obj":  _legacy_obj,
        "gltf": _legacy_gltf,
    }

# Explicit legacy configurations policy: add config names here to hide behind legacy button.
# Remains empty by default; user specifies which configs are legacy (not auto-detected by name).
LEGACY_CONFIGS = set()  # e.g., {"Assembly 1 (Legacy)"}  — add names here to hide them in a 'Legacy' category

# Active configuration — defaults to first available; overwritten by selector
_config_names = list(THRUSTER_CONFIGS.keys())
ACTIVE_CONFIG_NAME = _config_names[0] if _config_names else "Assembly 1 (Legacy)"

# Mesh files (set from active config; may be overridden by choose_thruster_config)
if THRUSTER_CONFIGS:
    OBJ_FILE  = THRUSTER_CONFIGS[ACTIVE_CONFIG_NAME]["obj"]
    GLTF_FILE = THRUSTER_CONFIGS[ACTIVE_CONFIG_NAME]["gltf"]
else:
    OBJ_FILE  = os.path.join(DATA_DIR, "Assembly 1.obj")
    GLTF_FILE = os.path.join(DATA_DIR, "Assembly 1.gltf")

MESH_SCALE = (1.0, 1.0, 1.0)

# Rotate body so the mesh faces the right direction in the sim.
# New CAD: -Y is "forward" in mesh space (camera at Y=-0.38).
# Rotate +90° about Z: mesh -Y → sim +X (forward)
MESH_BODY_EULER_DEG = (0.0, 0.0, 90.0)

AUTO_DETECT_THRUSTERS = True  # uses GLTF nodes if available

# ============================================
# RENDERING OPTIMIZATION
# ============================================
ENABLE_CAMERA_PREVIEW = True   # ROV on-board camera rendered to panel window
ENABLE_THRUSTER_ARROWS = True  # Visual thrust direction indicators (cone bodies, zero-lag)
ENABLE_MARKERS = False  # Thruster position marker spheres (expensive)
ENABLE_JOYSTICK_PANEL = True   # Open virtual joystick window on startup

# Joystick thruster switching cooldown — thrusters cannot reverse direction
# faster than this interval (seconds), matching real ESC response time.
JOYSTICK_SWITCH_COOLDOWN = 0.5

# Camera servo: the on-board camera is mounted on a single-axis tilt servo.
# The tilt angle is controlled by the slider on the joystick panel (-90°..+90°).
CAMERA_SERVO_MIN_DEG = -90.0   # max tilt down
CAMERA_SERVO_MAX_DEG =  90.0   # max tilt up

# Camera defaults + keyboard steps
CAM_DIST = 1.5      # Closer for small ROV
CAM_YAW = 0         # Look straight at ROV from ahead
CAM_PITCH = -25     # Look slightly down at ROV
CAM_TARGET = [0.2, 0.0, 0.4]  # Point ahead of ROV towards obstacles
CAM_STEP_ANGLE = 3.0
CAM_STEP_DIST = 0.10
CAM_STEP_PAN = 0.05

# Auto-follow the ROV so it stays in view
CAM_FOLLOW = True
CAM_FOLLOW_Z_OFFSET = 0.15  # Increased for better view of ROV top

# ROV camera preview (rendered from a camera node in the GLTF if present)
CAM_PREVIEW_W = 320
CAM_PREVIEW_H = 240
CAM_FOV = 70.0
CAM_NEAR = 0.02
CAM_FAR = 15.0
PREVIEW_FPS = 30  # 30 fps camera feed for smooth video
VIS_FPS = 60      # Marker/overlay update rate (Hz) — full rate for smooth arrows

# Camera renderer — ER_TINY_RENDERER (software) is safest across all platforms.
# ER_BULLET_HARDWARE_OPENGL is faster but segfaults on macOS Apple Silicon.
CAM_RENDERER = p.ER_TINY_RENDERER

# ── Screen Recording ────────────────────────────────────────────────
# Captures the PyBullet 3D view + full controller panel side-by-side as MP4.
# Triggered by the ⏺ REC button on the controller panel.
REC_FPS         = 20        # recording frame rate (lower than preview to reduce load)
REC_3D_W        = 640       # width of the 3D view capture
REC_3D_H        = 480       # height of the 3D view capture
# Controller panel frame is CTRL_W×CTRL_H (720×480) from joystick_panel
REC_SAVE_DIR    = os.path.expanduser("~/Downloads")  # where MP4s are saved

# Backwards thrust scaling: DDR says 4.45N reverse vs 5.56N forward ≈ 0.80
BACKWARDS_THRUST_SCALE = 0.80

# ============================================
# ENVIRONMENT PRESETS
# ============================================
# Each preset defines water/current/pool parameters.  The active preset is
# applied at startup (and can be changed via the config selector).
ENVIRONMENT_PRESETS = {
    "pool": {
        "label": "Pool / Test Tank",
        "surface_z": 1.2,
        "seabed_z": -2.0,
        "pool_half_x": 3.0,
        "pool_half_y": 2.0,
        "current_base": (0.01, 0.005, 0.0),
        "current_var_amp": (0.005, 0.003, 0.002),
        "current_var_period": (15.0, 20.0, 25.0),
        "fog_depth_range": 20.0,
        "fog_color_surface": (0.15, 0.45, 0.55),
        "fog_color_deep": (0.04, 0.08, 0.12),
        "num_risers": 3,
        "num_obstacles": 0,
        "description": "Calm indoor pool — minimal current, clear visibility",
    },
    "harbor": {
        "label": "Harbor / Dock",
        "surface_z": 2.0,
        "seabed_z": -6.0,
        "pool_half_x": 8.0,
        "pool_half_y": 6.0,
        "current_base": (0.05, 0.03, 0.0),
        "current_var_amp": (0.03, 0.02, 0.005),
        "current_var_period": (10.0, 14.0, 18.0),
        "fog_depth_range": 10.0,
        "fog_color_surface": (0.12, 0.35, 0.40),
        "fog_color_deep": (0.03, 0.06, 0.08),
        "num_risers": 5,
        "num_obstacles": 3,
        "description": "Moderate current, reduced visibility, pilings & debris",
    },
    "deep_ocean": {
        "label": "Deep Ocean",
        "surface_z": 5.0,
        "seabed_z": -30.0,
        "pool_half_x": 20.0,
        "pool_half_y": 20.0,
        "current_base": (0.08, 0.04, 0.01),
        "current_var_amp": (0.06, 0.04, 0.02),
        "current_var_period": (8.0, 12.0, 16.0),
        "fog_depth_range": 6.0,
        "fog_color_surface": (0.08, 0.25, 0.35),
        "fog_color_deep": (0.01, 0.03, 0.05),
        "num_risers": 0,
        "num_obstacles": 0,
        "description": "Deep water — strong current, poor visibility at depth",
    },
    "strong_current": {
        "label": "Strong Current",
        "surface_z": 1.5,
        "seabed_z": -4.0,
        "pool_half_x": 6.0,
        "pool_half_y": 4.0,
        "current_base": (0.15, 0.08, 0.0),
        "current_var_amp": (0.10, 0.06, 0.01),
        "current_var_period": (6.0, 8.0, 12.0),
        "fog_depth_range": 12.0,
        "fog_color_surface": (0.10, 0.35, 0.45),
        "fog_color_deep": (0.03, 0.06, 0.10),
        "num_risers": 2,
        "num_obstacles": 2,
        "description": "Challenging conditions — tests station-keeping ability",
    },
    "training": {
        "label": "Training",
        "surface_z": 1.0,
        "seabed_z": -1.5,
        "pool_half_x": 2.5,
        "pool_half_y": 2.0,
        "current_base": (0.0, 0.0, 0.0),
        "current_var_amp": (0.0, 0.0, 0.0),
        "current_var_period": (30.0, 30.0, 30.0),
        "fog_depth_range": 30.0,
        "fog_color_surface": (0.18, 0.50, 0.60),
        "fog_color_deep": (0.10, 0.20, 0.30),
        "num_risers": 0,
        "num_obstacles": 0,
        "description": "No current, shallow, max visibility — learn the controls",
    },
}
ACTIVE_ENVIRONMENT = "pool"  # default preset key

# ============================================
# ASSIST MODES (operator control aids)
# ============================================
# Depth hold: PD controller that holds current depth when engaged
DEPTH_HOLD_ENABLED = False
DEPTH_HOLD_KP = 15.0       # N per m of depth error
DEPTH_HOLD_KD = 8.0        # N per (m/s) of vertical velocity
DEPTH_HOLD_TARGET = None    # set by toggle key (captures current depth)
DEPTH_HOLD_MAX_FORCE = 12.0 # N — clamp to prevent runaway

# Heading hold: PD controller that holds current heading when engaged
HEADING_HOLD_ENABLED = False
HEADING_HOLD_KP = 4.0      # N·m per rad of heading error
HEADING_HOLD_KD = 2.0      # N·m per (rad/s) of yaw rate
HEADING_HOLD_TARGET = None  # set by toggle key (captures current heading)
HEADING_HOLD_MAX_TORQUE = 5.0  # N·m — clamp

# Station keeping: depth hold + heading hold combined
# (activated separately or via a combined toggle)

# ============================================
# INPUT CURVES & JOYSTICK CONFIGURATION
# ============================================
# Proportional mode: when True, joystick outputs continuous -1..+1 instead
# of snapping to binary -1/0/+1.  Thruster spool dynamics still apply.
PROPORTIONAL_MODE = False

# Input curve exponent: >1 gives more precision near centre, <1 more at edges
# Applied as: output = sign(input) * |input|^exponent
INPUT_CURVE_EXPONENT = 1.5

# Deadzone: stick displacement below this fraction is treated as zero
INPUT_DEADZONE = 0.15

# Rate limit: maximum change in command per second (0=unlimited)
INPUT_RATE_LIMIT = 0.0  # commands/sec (0 = no limit, instant)

# ============================================
# TIMING / PERFORMANCE METRICS
# ============================================
SHOW_TIMING_METRICS = True     # print FPS / step-time periodically
TIMING_REPORT_INTERVAL = 5.0   # seconds between timing reports

# ============================================
# THRUSTER FAILURE SIMULATION
# ============================================
THRUSTER_FAILURE_ENABLED = False
THRUSTER_FAILURE_PROB = 0.0     # probability per second of a thruster failing
THRUSTER_FAILED = []            # list of bools, set at runtime
THRUSTER_FAILURE_DURATION = 0.0 # seconds (0 = permanent until reset)

# ============================================
# DEBUG VISUALIZATION
# ============================================
SHOW_FORCE_VECTORS = False   # draw thrust/drag/buoyancy vectors in 3D view
FORCE_VECTOR_SCALE = 0.05    # meters per Newton for visualization

# Obstacles (movable props) — also treated as neutrally-buoyant objects in water
NUM_OBSTACLES = 0                          # Set to 0 to disable random objects
OBSTACLE_SPAWN_CENTER = (0.8, 0.0, 0.40)  # Ahead of ROV start
OBSTACLE_SPREAD = (0.4, 0.5, 0.15)        # +/- spread around center
OBSTACLE_MASS = 0.5
OBSTACLE_SIZE_RANGE = (0.03, 0.08)        # box half-extent range (m) — smaller for small ROV

# Make obstacles "float" in water (neutral buoyancy) and experience drag
OBSTACLE_BUOYANCY_SCALE = 1.03   # slightly positive so they don't slowly sink
OBSTACLE_DRAG_LIN = 3.0          # N per (m/s) — scaled for lighter obstacles
OBSTACLE_DRAG_QUAD = 5.0         # N per (m/s)^2
OBSTACLE_MAX_DRAG = 30.0         # clamp for stability

# Terminal debug logging (prints state regularly)
LOG_FPS = 0                      # 0 = disabled (console I/O causes lag)
LOG_OBS = False                  # set True to show obstacle positions in telemetry

# Detailed physics logging (structured, CSV-like lines written to the same log file)
LOG_PHYSICS_DETAILED = False    # set True for diagnostics (writes per-step CSV to log file)
LOG_PHYSICS_HZ = 10            # how many detailed lines per second (approx)

# Marker spheres
MARKER_RADIUS = 0.045
MARKER_OFFSET = 0.00  # keep 0 to sit right on the CAD location
COLOR_OFF = [1.0, 0.15, 0.15, 0.0]  # OFF = invisible
COLOR_ON  = [0.15, 1.0, 0.15, 1.0]  # green

# Fallback thrusters if auto-detect fails (body frame: x forward, y left, z up)
# DDR: 3 horizontal + 1 vertical, rear two angled at 45°
_c45 = 0.7071  # cos(45°) = sin(45°)
THRUSTERS = [
    {"name":"thruster_1", "pos": (-0.16,  0.13, -0.00), "dir": (_c45, _c45, 0), "key": ord('1'), "kind":"H"},  # rear-left, 45°
    {"name":"thruster_2", "pos": ( 0.16,  0.13, -0.01), "dir": (_c45,-_c45, 0), "key": ord('2'), "kind":"H"},  # rear-right, 45°
    {"name":"thruster_3", "pos": ( 0.00,  0.00, -0.03), "dir": (0, 0, 1),       "key": ord('3'), "kind":"V"},  # vertical (centre)
    {"name":"thruster_4", "pos": ( 0.00, -0.24,  0.00), "dir": (0,-1, 0),       "key": ord('4'), "kind":"H"},  # front horizontal
]


# ==========================
# UTILx
# ==========================
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def deg2rad(d):
    return d * math.pi / 180.0

def vmag(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def vnorm(v):
    n = vmag(v) or 1.0
    return (v[0]/n, v[1]/n, v[2]/n)

def vdot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def vcross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])

def _safe_camera_rgba(raw_pixels, width, height):
    """Convert PyBullet getCameraImage pixel data to a (H, W, 4) uint8 numpy array.

    On macOS/Linux with numpy, PyBullet returns a numpy array directly.
    On Windows (or without numpy), it may return a flat tuple, list, or bytes
    which cannot be indexed with [:, :, :3].  This helper handles all cases.
    Returns None if conversion fails.
    """
    if raw_pixels is None:
        return None
    if not HAS_NUMPY:
        return None
    try:
        if isinstance(raw_pixels, np.ndarray):
            if raw_pixels.ndim == 3 and raw_pixels.shape == (height, width, 4):
                return raw_pixels
            # Might be flat — try reshape
            return raw_pixels.reshape(height, width, 4).astype(np.uint8)
        # tuple, list, or bytes — convert to array then reshape
        arr = np.array(raw_pixels, dtype=np.uint8)
        return arr.reshape(height, width, 4)
    except (ValueError, TypeError, AttributeError):
        return None

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
    # Parse vertices and face connectivity so we can ignore disconnected mesh
    # components that are far from the main ROV body (common CAD export artifact).
    vertices = []
    face_vertex_lists = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    except ValueError:
                        continue
            elif line.startswith("f "):
                tokens = line.strip().split()[1:]
                face_idxs = []
                for tok in tokens:
                    vtok = tok.split("/")[0]
                    if not vtok:
                        continue
                    try:
                        idx = int(vtok)
                    except ValueError:
                        continue
                    if idx < 0:
                        idx = len(vertices) + idx + 1
                    if 1 <= idx <= len(vertices):
                        face_idxs.append(idx - 1)
                if len(face_idxs) >= 3:
                    face_vertex_lists.append(face_idxs)

    if not vertices:
        center = (0.0, 0.0, 0.0)
        size = (0.4, 0.4, 0.25)
        return center, size

    # If we have no faces, fall back to all-vertex bounds.
    if not face_vertex_lists:
        vmin = [1e9, 1e9, 1e9]
        vmax = [-1e9, -1e9, -1e9]
        for x, y, z in vertices:
            vmin[0] = min(vmin[0], x); vmin[1] = min(vmin[1], y); vmin[2] = min(vmin[2], z)
            vmax[0] = max(vmax[0], x); vmax[1] = max(vmax[1], y); vmax[2] = max(vmax[2], z)
        center = ((vmin[0]+vmax[0])/2, (vmin[1]+vmax[1])/2, (vmin[2]+vmax[2])/2)
        size = (vmax[0]-vmin[0], vmax[1]-vmin[1], vmax[2]-vmin[2])
        return center, size

    # Union-find over face-connected vertices.
    parent = {}
    rank = {}

    def uf_make(i):
        if i not in parent:
            parent[i] = i
            rank[i] = 0

    def uf_find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def uf_union(a, b):
        ra = uf_find(a)
        rb = uf_find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for face in face_vertex_lists:
        base = face[0]
        uf_make(base)
        for vi in face[1:]:
            uf_make(vi)
            uf_union(base, vi)

    components = {}
    for vi in parent:
        root = uf_find(vi)
        components.setdefault(root, []).append(vi)

    # Compute per-component stats.
    comp_stats = []
    for _, idxs in components.items():
        vmin = [1e9, 1e9, 1e9]
        vmax = [-1e9, -1e9, -1e9]
        for vi in idxs:
            x, y, z = vertices[vi]
            vmin[0] = min(vmin[0], x); vmin[1] = min(vmin[1], y); vmin[2] = min(vmin[2], z)
            vmax[0] = max(vmax[0], x); vmax[1] = max(vmax[1], y); vmax[2] = max(vmax[2], z)
        cx = (vmin[0] + vmax[0]) / 2
        cy = (vmin[1] + vmax[1]) / 2
        cz = (vmin[2] + vmax[2]) / 2
        sx = (vmax[0] - vmin[0])
        sy = (vmax[1] - vmin[1])
        sz = (vmax[2] - vmin[2])
        diag = math.sqrt(sx*sx + sy*sy + sz*sz)
        comp_stats.append({
            "idxs": idxs,
            "count": len(idxs),
            "center": (cx, cy, cz),
            "diag": diag,
            "vmin": vmin,
            "vmax": vmax,
        })

    # Main body = largest face-connected component.
    comp_stats.sort(key=lambda c: c["count"], reverse=True)
    main = comp_stats[0]
    main_c = main["center"]
    keep_components = [main]

    # Keep nearby components; drop far-out disconnected ones that skew origin.
    # Threshold scales with main-body size, with a small absolute floor.
    keep_dist = max(0.75, 2.5 * max(main["diag"], 1e-6))
    dropped = 0
    for comp in comp_stats[1:]:
        c = comp["center"]
        dist = math.sqrt((c[0]-main_c[0])**2 + (c[1]-main_c[1])**2 + (c[2]-main_c[2])**2)
        if dist <= keep_dist:
            keep_components.append(comp)
        else:
            dropped += 1

    vmin = [1e9, 1e9, 1e9]
    vmax = [-1e9, -1e9, -1e9]
    kept_vertex_count = 0
    for comp in keep_components:
        cmin = comp["vmin"]
        cmax = comp["vmax"]
        kept_vertex_count += comp["count"]
        vmin[0] = min(vmin[0], cmin[0]); vmin[1] = min(vmin[1], cmin[1]); vmin[2] = min(vmin[2], cmin[2])
        vmax[0] = max(vmax[0], cmax[0]); vmax[1] = max(vmax[1], cmax[1]); vmax[2] = max(vmax[2], cmax[2])

    # Robustly trim the outer tails so sparse far geometry does not skew the
    # center/collision box, while preserving the dense main frame.
    TRIM_FRAC = 0.04
    MIN_VERTS_FOR_TRIM = 5000
    if kept_vertex_count >= MIN_VERTS_FOR_TRIM:
        xs = []
        ys = []
        zs = []
        for comp in keep_components:
            for vi in comp["idxs"]:
                x, y, z = vertices[vi]
                xs.append(x)
                ys.append(y)
                zs.append(z)

        xs.sort()
        ys.sort()
        zs.sort()

        n = len(xs)
        lo = int(n * TRIM_FRAC)
        hi = n - lo - 1
        if 0 <= lo < hi < n:
            tvmin = [xs[lo], ys[lo], zs[lo]]
            tvmax = [xs[hi], ys[hi], zs[hi]]

            raw_center = ((vmin[0] + vmax[0]) / 2, (vmin[1] + vmax[1]) / 2, (vmin[2] + vmax[2]) / 2)
            med_center = (xs[n // 2], ys[n // 2], zs[n // 2])
            raw_size = (max(vmax[0] - vmin[0], 1e-9), max(vmax[1] - vmin[1], 1e-9), max(vmax[2] - vmin[2], 1e-9))
            center_shift = (
                abs(raw_center[0] - med_center[0]),
                abs(raw_center[1] - med_center[1]),
                abs(raw_center[2] - med_center[2]),
            )

            # Trigger trimming only when one axis is clearly skewed by a tail.
            skewed = any(
                (center_shift[i] > 0.02) and ((center_shift[i] / raw_size[i]) > 0.10)
                for i in range(3)
            )

            # Use trimmed bounds only if they remain physically close to raw
            # bounds size (avoid over-shrinking tiny or pathological meshes).
            raw_sx = max(vmax[0] - vmin[0], 1e-9)
            raw_sy = max(vmax[1] - vmin[1], 1e-9)
            raw_sz = max(vmax[2] - vmin[2], 1e-9)
            trim_sx = max(tvmax[0] - tvmin[0], 0.0)
            trim_sy = max(tvmax[1] - tvmin[1], 0.0)
            trim_sz = max(tvmax[2] - tvmin[2], 0.0)

            if skewed and (trim_sx / raw_sx) > 0.55 and (trim_sy / raw_sy) > 0.55 and (trim_sz / raw_sz) > 0.55:
                vmin = tvmin
                vmax = tvmax

    if dropped > 0:
        print(f"[INFO] Ignored {dropped} far disconnected mesh component(s) in {os.path.basename(path)}.")

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

    Position comes from the "Lense" child (camera aperture), while orientation
    comes from the "Head" child (camera housing look-direction).  A small
    forward offset is added so the virtual camera sits just outside the hull.
    """
    if not os.path.exists(gltf_path):
        return None
    try:
        with open(gltf_path, "r", errors="ignore") as f:
            gltf = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
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

    # Find orientation from "Head" child and position from "Lense" child.
    cam_root = cam_candidates[0]
    children = nodes[cam_root].get("children", []) or []

    head_idx = None
    lense_idx = None
    for child_idx in children:
        child_nm = (nodes[child_idx].get("name", "") or "").lower()
        if "head" in child_nm:
            head_idx = child_idx
        if "lense" in child_nm or "lens" in child_nm:
            lense_idx = child_idx

    # Orientation from Head (camera housing look-direction)
    orient_idx = head_idx
    if orient_idx is None:
        for child_idx in children:
            child_nm = (nodes[child_idx].get("name", "") or "").lower()
            if "body" in child_nm and not is_identity_trs(nodes[child_idx]):
                orient_idx = child_idx
                break
    if orient_idx is None:
        orient_idx = find_first_transform_descendant(nodes, cam_root)

    m = world_mat(orient_idx)
    c0, c1, c2, t = gltf_mat_basis_and_pos(m)

    # glTF convention: -Z is forward, +Y is up.
    forward = (-c2[0], -c2[1], -c2[2])
    up = c1

    # Position from Lense child (camera aperture) if available, else Head
    if lense_idx is not None:
        m_pos = world_mat(lense_idx)
        _, _, _, t_pos = gltf_mat_basis_and_pos(m_pos)
    else:
        t_pos = t

    pos_body = (t_pos[0] - center[0], t_pos[1] - center[1], t_pos[2] - center[2])

    # The GLTF camera orientation from Fusion360 is unreliable (Head node's -Z
    # doesn't point along the vehicle's forward axis).  Override forward to be
    # the mesh's forward direction: -Y in mesh space.  Up stays +Z.
    forward = (0.0, -1.0, 0.0)
    up      = (0.0,  0.0, 1.0)

    # Nudge position along forward so the virtual camera sits outside the hull
    # and clears the visible gear/mechanism (avoids clipping into ROV mesh).
    CAM_FWD_NUDGE = 0.10  # 10 cm forward — enough to clear camera housing
    fwd_n = vnorm(forward)
    pos_body = (pos_body[0] + fwd_n[0] * CAM_FWD_NUDGE,
                pos_body[1] + fwd_n[1] * CAM_FWD_NUDGE,
                pos_body[2] + fwd_n[2] * CAM_FWD_NUDGE)

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
    except (OSError, json.JSONDecodeError, ValueError):
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
            # The GLTF Y-axis (thrust direction) for Fusion-exported thrusters
            # points away from camera-forward (mesh -Y) for this model.
            # Negate ALL horizontal thrusters so cmd=+1 means "push forward."
            # This preserves the left/right asymmetry of angled rear thrusters
            # (T1/T2), unlike the old heuristic that flipped based on X sign
            # (which broke T2's direction and made yaw impossible).
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
# CONFIGURATION SELECTOR
# ==========================

# ── Color palette (dark professional theme) ──────────────────────
_CS = {
    "bg":       "#0f1724",
    "card":     "#162032",
    "card_bd":  "#2a3a52",
    "canvas":   "#0a1018",
    "grid":     "#131c28",
    "body":     "#1e2d42",
    "body_bd":  "#3d5a80",
    "frame":    "#243447",
    "thr_h":    "#f59e0b",
    "thr_h_bg": "#2d1a00",
    "thr_v":    "#3b82f6",
    "thr_v_bg": "#0f2744",
    "fwd":      "#10b981",
    "angle":    "#ef4444",
    "text":     "#e2e8f0",
    "dim":      "#7a8a9e",
    "btn":      "#2563eb",
    "btn_hvr":  "#1d4ed8",
    "btn_txt":  "#ffffff",
    "launch_bg": "#ffb703",
    "launch_bg_hvr": "#ffcb47",
    "launch_txt": "#1f2937",
    "legacy_bg": "#334155",
    "legacy_bg_hvr": "#475569",
    "legacy_txt": "#e2e8f0",
    "label":    "#94a3b8",
    "nose":     "#29584d",
    "com":      "#64748b",
    "com_bd":   "#94a3b8",
    "sep":      "#1e2d3d",
    "stat_v":   "#cbd5e1",
}


def _cfg_draw_schematic(canvas, info, cw, ch):
    """Draw a top-down ROV schematic with thruster vectors on a tk.Canvas."""
    C = _CS
    thrusters = info["thrusters"]
    size = info["size"]

    # ── Coordinate mapping: mesh → screen ──
    # Mesh: X=right, -Y=forward, Z=up.  Screen: X=right, Y=down.
    # So screen_x = mesh_x, screen_y = mesh_y (positive mesh Y = rear = down).
    margin = 58
    usable = min(cw, ch) - 2 * margin
    max_ext = max(size[0], size[1], 0.2)
    scale = usable / max_ext * 0.62
    ox, oy = cw / 2, ch / 2 + 12  # offset down a bit for FWD label room

    def m2s(mx, my):
        return ox + mx * scale, oy + my * scale

    # ── Grid ──
    gs = 28
    for gx in range(0, cw + gs, gs):
        canvas.create_line(gx, 0, gx, ch, fill=C["grid"])
    for gy in range(0, ch + gs, gs):
        canvas.create_line(0, gy, cw, gy, fill=C["grid"])

    # ── ROV body outline ──
    hx = size[0] / 2 * scale
    hy = size[1] / 2 * scale

    # Tapered nose polygon for forward end
    nose_inset = hx * 0.30
    nose_ext   = hy * 0.18
    body_pts = [
        ox,            oy - hy - nose_ext,   # nose tip (front)
        ox + hx - nose_inset, oy - hy,       # front-right shoulder
        ox + hx,              oy - hy * 0.5,  # upper-right
        ox + hx,              oy + hy,        # rear-right
        ox - hx,              oy + hy,        # rear-left
        ox - hx,              oy - hy * 0.5,  # upper-left
        ox - hx + nose_inset, oy - hy,       # front-left shoulder
    ]
    canvas.create_polygon(body_pts, fill=C["body"], outline=C["body_bd"],
                          width=2, smooth=False)

    # Internal frame cross-members (visual detail)
    for frac in (-0.35, 0.0, 0.35):
        y = oy + frac * hy * 2
        canvas.create_line(ox - hx * 0.75, y, ox + hx * 0.75, y,
                           fill=C["frame"], width=1, dash=(4, 4))
    canvas.create_line(ox, oy - hy * 0.45, ox, oy + hy * 0.85,
                       fill=C["frame"], width=1, dash=(4, 4))

    # ── Forward direction indicator ──
    fwd_base_y = oy - hy - nose_ext - 4
    fwd_tip_y  = fwd_base_y - 28
    canvas.create_line(ox, fwd_base_y, ox, fwd_tip_y, fill=C["fwd"],
                       width=2, arrow="last", arrowshape=(8, 10, 4))
    canvas.create_text(ox, fwd_tip_y - 11, text="FWD", fill=C["fwd"],
                       font=("Helvetica", 9, "bold"))

    # ── Centre of mass crosshair ──
    cr = 5
    canvas.create_line(ox - cr, oy, ox + cr, oy, fill=C["com_bd"], width=1)
    canvas.create_line(ox, oy - cr, ox, oy + cr, fill=C["com_bd"], width=1)
    canvas.create_oval(ox - 3, oy - 3, ox + 3, oy + 3,
                       fill=C["com"], outline=C["com_bd"])

    # ── "REAR" label at the bottom of body ──
    canvas.create_text(ox, oy + hy + 14, text="REAR", fill=C["dim"],
                       font=("Helvetica", 7))

    # ── Thrusters ──
    for t in thrusters:
        px, py = m2s(t["pos"][0], t["pos"][1])
        tnum = t["name"][-1]  # "1"–"4"

        if t["kind"] == "V":
            # Vertical thruster: circle + heave arrow icon
            vr = 11
            canvas.create_oval(px - vr, py - vr, px + vr, py + vr,
                               fill=C["thr_v_bg"], outline=C["thr_v"], width=2)
            # up-arrow glyph drawn manually (more reliable than unicode)
            canvas.create_line(px, py + 4, px, py - 5, fill=C["thr_v"],
                               width=2, arrow="last", arrowshape=(5, 6, 3))
            canvas.create_text(px, py + vr + 12, text=f"T{tnum}",
                               fill=C["thr_v"], font=("Helvetica", 9, "bold"))
            canvas.create_text(px, py + vr + 24, text="HEAVE",
                               fill=C["dim"], font=("Helvetica", 7))
        else:
            # Horizontal thruster: position dot + thrust direction arrow
            hr = 7
            canvas.create_oval(px - hr, py - hr, px + hr, py + hr,
                               fill=C["thr_h_bg"], outline=C["thr_h"], width=2)

            # Thrust direction arrow
            arrow_len = 55
            dx = t["dir"][0] * arrow_len
            dy = t["dir"][1] * arrow_len
            canvas.create_line(px, py, px + dx, py + dy,
                               fill=C["thr_h"], width=3, arrow="last",
                               arrowshape=(10, 13, 5))

            # Thruster label (offset away from body centre)
            is_left  = t["pos"][0] < -0.03
            is_right = t["pos"][0] > 0.03
            if is_left:
                lx, anc = px - hr - 7, "e"
            elif is_right:
                lx, anc = px + hr + 7, "w"
            else:
                lx, anc = px + hr + 7, "w"
            canvas.create_text(lx, py, text=f"T{tnum}",
                               fill=C["thr_h"], font=("Helvetica", 9, "bold"),
                               anchor=anc)

            # ── Angle arc annotation (only for off-centreline thrusters) ──
            if abs(t["pos"][0]) > 0.05:
                # Angle between thrust direction and forward (-Y in mesh)
                # Using tkinter angle convention: 0°=east, 90°=north, CCW+
                arrow_tk_angle = math.degrees(math.atan2(-dy, dx))
                angle_diff = arrow_tk_angle - 90.0  # deviation from forward (90°)
                abs_angle = abs(angle_diff)

                if abs_angle > 2.0:
                    arc_r = 28
                    # Arc from forward (90°) sweeping toward arrow direction
                    if angle_diff > 0:  # left side
                        start_a = 90.0
                        extent_a = angle_diff
                    else:               # right side
                        start_a = 90.0
                        extent_a = angle_diff  # negative = clockwise

                    canvas.create_arc(
                        px - arc_r, py - arc_r, px + arc_r, py + arc_r,
                        start=start_a, extent=extent_a,
                        style="arc", outline=C["angle"], width=2)

                    # Dashed reference line straight "up" from thruster (forward ref)
                    canvas.create_line(px, py, px, py - arc_r - 4,
                                       fill=C["dim"], width=1, dash=(3, 3))

                    # Angle value label at midpoint of arc
                    mid_rad = math.radians(90.0 + angle_diff / 2)
                    lr = arc_r + 16
                    lx2 = px + lr * math.cos(mid_rad)
                    ly2 = py - lr * math.sin(mid_rad)
                    canvas.create_text(lx2, ly2, text=f"{abs_angle:.1f}\u00b0",
                                       fill=C["angle"],
                                       font=("Helvetica", 10, "bold"))


def choose_thruster_config():
    """
    Professional Tkinter configuration selector with schematic renderings
    of each thruster layout, highlighting angle differences and key metrics.
    """
    global OBJ_FILE, GLTF_FILE, ACTIVE_CONFIG_NAME

    if len(THRUSTER_CONFIGS) <= 1:
        if THRUSTER_CONFIGS:
            name = list(THRUSTER_CONFIGS.keys())[0]
            OBJ_FILE  = THRUSTER_CONFIGS[name]["obj"]
            GLTF_FILE = THRUSTER_CONFIGS[name]["gltf"]
            ACTIVE_CONFIG_NAME = name
            return name
        return ACTIVE_CONFIG_NAME

    import tkinter as tk

    # ── Pre-analyse each configuration (with exception protection) ────────────────────────────
    config_info = {}
    for name, cfg in THRUSTER_CONFIGS.items():
        try:
            center, size = obj_bounds(cfg["obj"])
            thrusters = detect_thrusters_from_gltf(cfg["gltf"], center)
            h_thrs = [t for t in thrusters if t["kind"] == "H"]

            # Outboard angles for angled rear thrusters (off-centreline)
            rear_angles = []
            for t in h_thrs:
                if abs(t["pos"][0]) > 0.05:
                    rear_angles.append(
                        math.degrees(math.atan2(abs(t["dir"][0]), abs(t["dir"][1]))))
            avg_angle = sum(rear_angles) / len(rear_angles) if rear_angles else 0.0

            # Total forward thrust when all H thrusters fire at 100%
            total_fwd = sum(abs(t["dir"][1]) for t in h_thrs) * MAX_THRUST_H

            # Yaw torque estimate: lateral force component × moment arm
            yaw_torque = 0.0
            for t in h_thrs:
                if abs(t["pos"][0]) > 0.05:
                    yaw_torque += abs(t["dir"][0]) * MAX_THRUST_H * abs(t["pos"][1])

            # Descriptive tag
            if avg_angle > 30:
                desc = "Wide-angle layout — strong yaw authority,\nmoderate forward thrust"
            elif avg_angle > 15:
                desc = "Narrow-angle layout — maximum forward\nthrust, moderate yaw authority"
            else:
                desc = "Straight layout — pure forward thrust,\nminimal yaw authority"

            config_info[name] = {
                "thrusters":     thrusters,
                "size":          size,
                "total_fwd_N":   total_fwd,
                "avg_angle":     avg_angle,
                "yaw_torque_Nm": yaw_torque,
                "n_h":           len(h_thrs),
                "n_v":           len([t for t in thrusters if t["kind"] == "V"]),
                "description":   desc,
            }
        except (ValueError, KeyError, pybullet.error) as e:
            print(f"⚠️  Pre-analysis of '{name}' failed: {e} — skipping from selector")
            config_info[name] = None  # Mark as failed to skip rendering

    C = _CS
    # Split configs using explicit LEGACY_CONFIGS policy (manual, not name-based)
    primary_configs = [(name, cfg) for name, cfg in THRUSTER_CONFIGS.items() if name not in LEGACY_CONFIGS and config_info.get(name) is not None]
    legacy_configs = [(name, cfg) for name, cfg in THRUSTER_CONFIGS.items() if name in LEGACY_CONFIGS and config_info.get(name) is not None]

    # Fallback: if no primary configs due to all being legacy OR all failing analysis, show all non-failed
    if not primary_configs:
        primary_configs = [(name, cfg) for name, cfg in THRUSTER_CONFIGS.items() if config_info.get(name) is not None]

    chosen = [None]

    # ── Window ───────────────────────────────────────────────────
    root = tk.Tk()
    root.title("ROV Simulator \u2014 Thruster Configuration")
    root.configure(bg=C["bg"])
    root.resizable(False, False)

    n = len(primary_configs)
    CARD_W   = 310 if n <= 3 else 260
    CANVAS_H = 280
    PAD      = 14
    WIN_PAD  = 22
    WIN_W    = n * CARD_W + (n + 1) * PAD + 2 * WIN_PAD
    WIN_H    = 600
    sx = root.winfo_screenwidth()  // 2 - WIN_W // 2
    sy = root.winfo_screenheight() // 2 - WIN_H // 2
    root.geometry(f"{WIN_W}x{WIN_H}+{sx}+{sy}")

    # ── Title bar ────────────────────────────────────────────────
    hdr = tk.Frame(root, bg=C["bg"])
    hdr.pack(fill="x", padx=WIN_PAD, pady=(WIN_PAD, 0))
    tk.Label(hdr, text="Thruster Configuration",
             font=("Helvetica", 18, "bold"), fg=C["text"], bg=C["bg"]
             ).pack(side="left")
    subtitle = "Select a layout to simulate"
    if legacy_configs:
        subtitle += " — legacy layouts are in the button below"
    tk.Label(hdr, text=subtitle,
             font=("Helvetica", 11), fg=C["dim"], bg=C["bg"]
             ).pack(side="left", padx=(14, 0), pady=(5, 0))

    # Separator
    tk.Frame(root, height=1, bg=C["card_bd"]).pack(fill="x", padx=WIN_PAD,
                                                    pady=(10, 12))

    # ── Cards container ──────────────────────────────────────────
    cards = tk.Frame(root, bg=C["bg"])
    cards.pack(fill="both", expand=True, padx=WIN_PAD)

    def _pick(name):
        chosen[0] = name
        root.destroy()

    def _bind_hover(widget, base_bg, hover_bg):
        widget.bind("<Enter>", lambda _e: widget.configure(bg=hover_bg))
        widget.bind("<Leave>", lambda _e: widget.configure(bg=base_bg))

    def _bind_hover_pair(frame_widget, label_widget, base_bg, hover_bg):
        frame_widget.bind("<Enter>", lambda _e: (frame_widget.configure(bg=hover_bg), label_widget.configure(bg=hover_bg)))
        frame_widget.bind("<Leave>", lambda _e: (frame_widget.configure(bg=base_bg), label_widget.configure(bg=base_bg)))
        label_widget.bind("<Enter>", lambda _e: (frame_widget.configure(bg=hover_bg), label_widget.configure(bg=hover_bg)))
        label_widget.bind("<Leave>", lambda _e: (frame_widget.configure(bg=base_bg), label_widget.configure(bg=base_bg)))

    def _open_legacy_dialog():
        dlg = tk.Toplevel(root)
        dlg.title("Legacy Configurations")
        dlg.configure(bg=C["bg"])
        dlg.resizable(False, False)
        dlg.transient(root)
        dlg.grab_set()

        dk_w = 520
        dk_h = 80 + 64 * max(1, len(legacy_configs))
        dsx = dlg.winfo_screenwidth() // 2 - dk_w // 2
        dsy = dlg.winfo_screenheight() // 2 - dk_h // 2
        dlg.geometry(f"{dk_w}x{dk_h}+{dsx}+{dsy}")

        tk.Label(
            dlg,
            text="Choose a legacy layout",
            font=("Helvetica", 14, "bold"),
            fg=C["text"],
            bg=C["bg"],
        ).pack(anchor="w", padx=16, pady=(14, 6))

        if not legacy_configs:
            tk.Label(
                dlg,
                text="No legacy configurations found.",
                font=("Helvetica", 11),
                fg=C["dim"],
                bg=C["bg"],
            ).pack(anchor="w", padx=16, pady=(4, 10))
            return

        for name, _ in legacy_configs:
            row = tk.Frame(dlg, bg=C["bg"])
            row.pack(fill="x", padx=16, pady=6)

            btn = tk.Frame(row, bg=C["legacy_bg"], cursor="hand2", highlightthickness=0)
            btn.pack(fill="x")
            lbl = tk.Label(
                btn,
                text=f"Enter {name}",
                font=("Helvetica", 11, "bold"),
                fg=C["legacy_txt"],
                bg=C["legacy_bg"],
                padx=12,
                pady=8,
            )
            lbl.pack(fill="x")

            def _pick_legacy(_e=None, nm=name):
                chosen[0] = nm
                dlg.destroy()
                root.destroy()

            btn.bind("<Button-1>", _pick_legacy)
            lbl.bind("<Button-1>", _pick_legacy)
            _bind_hover_pair(btn, lbl, C["legacy_bg"], C["legacy_bg_hvr"])

    for col, (name, _) in enumerate(primary_configs):
        info = config_info.get(name)
        if info is None:
            continue  # Skip configs that failed pre-analysis

        # Card frame
        card = tk.Frame(cards, bg=C["card"], highlightbackground=C["card_bd"],
                        highlightthickness=1)
        card.grid(row=0, column=col, padx=PAD // 2, sticky="nsew")
        # inner padding frame
        inner = tk.Frame(card, bg=C["card"])
        inner.pack(fill="both", expand=True, padx=14, pady=12)

        # Config name
        tk.Label(inner, text=name, font=("Helvetica", 13, "bold"),
                 fg=C["text"], bg=C["card"]).pack(anchor="w")

        # Thin accent line under title
        tk.Frame(inner, height=2, bg=C["thr_h"]).pack(fill="x", pady=(4, 10))

        # ── Canvas (schematic) ──
        cv_w = CARD_W - 52
        cv = tk.Canvas(inner, width=cv_w, height=CANVAS_H,
                       bg=C["canvas"], highlightthickness=0, bd=0)
        cv.pack()
        _cfg_draw_schematic(cv, info, cv_w, CANVAS_H)

        # ── Stats rows ──
        sf = tk.Frame(inner, bg=C["card"])
        sf.pack(fill="x", pady=(12, 0))

        stats = [
            ("T1 / T2 Angle",  f"{info['avg_angle']:.1f}\u00b0 outboard"),
            ("Forward Thrust",  f"{info['total_fwd_N']:.1f} N"),
            ("Yaw Torque",      f"{info['yaw_torque_Nm']:.2f} N\u00b7m"),
            ("Thrusters",       f"{info['n_h']}H + {info['n_v']}V"),
        ]
        for lbl, val in stats:
            row = tk.Frame(sf, bg=C["card"])
            row.pack(fill="x", pady=1)
            tk.Label(row, text=lbl, font=("Helvetica", 10),
                     fg=C["label"], bg=C["card"]).pack(side="left")
            tk.Label(row, text=val, font=("Helvetica", 10, "bold"),
                     fg=C["stat_v"], bg=C["card"]).pack(side="right")

        # Description
        tk.Label(inner, text=info["description"],
                 font=("Helvetica", 9), fg=C["dim"], bg=C["card"],
                 justify="left", anchor="w").pack(fill="x", pady=(8, 0))

        # ── Select button (custom widget for consistent color on macOS) ──
        short = name.split()[-1]  # "V1", "V2", etc.
        launch_btn = tk.Frame(inner, bg=C["launch_bg"], cursor="hand2", highlightthickness=0)
        launch_btn.pack(fill="x", pady=(12, 0))
        launch_lbl = tk.Label(
            launch_btn,
            text=f"\u25b6   Launch {short}",
            font=("Helvetica", 12, "bold"),
            fg=C["launch_txt"],
            bg=C["launch_bg"],
            padx=8,
            pady=7,
        )
        launch_lbl.pack(fill="x")

        launch_btn.bind("<Button-1>", lambda _e, nm=name: _pick(nm))
        launch_lbl.bind("<Button-1>", lambda _e, nm=name: _pick(nm))
        _bind_hover_pair(launch_btn, launch_lbl, C["launch_bg"], C["launch_bg_hvr"])

    for i in range(n):
        cards.columnconfigure(i, weight=1)

    if legacy_configs:
        legacy_row = tk.Frame(root, bg=C["bg"])
        legacy_row.pack(fill="x", padx=WIN_PAD, pady=(8, 14))

        legacy_btn = tk.Frame(legacy_row, bg=C["legacy_bg"], cursor="hand2", highlightthickness=0)
        legacy_btn.pack(side="left")
        legacy_lbl = tk.Label(
            legacy_btn,
            text="Legacy Configurations",
            font=("Helvetica", 11, "bold"),
            fg=C["legacy_txt"],
            bg=C["legacy_bg"],
            padx=12,
            pady=8,
        )
        legacy_lbl.pack()

        legacy_btn.bind("<Button-1>", lambda _e: _open_legacy_dialog())
        legacy_lbl.bind("<Button-1>", lambda _e: _open_legacy_dialog())
        _bind_hover_pair(legacy_btn, legacy_lbl, C["legacy_bg"], C["legacy_bg_hvr"])

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.lift()
    try:
        root.attributes("-topmost", True)
    except (AttributeError, RuntimeError) as e:
        print(f"⚠️  Could not set window topmost attribute (macOS quirk): {e}")
    root.mainloop()

    if chosen[0] is None:
        return None

    cfg = THRUSTER_CONFIGS[chosen[0]]
    OBJ_FILE  = cfg["obj"]
    GLTF_FILE = cfg["gltf"]
    ACTIVE_CONFIG_NAME = chosen[0]
    print(f"[CONFIG] Selected: {chosen[0]}")
    print(f"[CONFIG]   OBJ:  {OBJ_FILE}")
    print(f"[CONFIG]   GLTF: {GLTF_FILE}")
    return chosen[0]


# ==========================
# PHYSICS
# ==========================

# NOTE: Buoyancy is applied INLINE in the main loop with depth-dependent adjustment.
# The standalone apply_buoyancy() function was removed to avoid confusion.

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

def apply_righting_torque(body_id, base_quat, ang_world, submersion=1.0):
    """
    Restore roll/pitch toward 0 using PD control (buoyancy-like righting).
    Prevents oscillation and excessive tilting.
    Scaled by submersion factor so it fades as ROV breaches the surface.
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

    # Scale by submersion: no righting moment above water
    tx_b *= submersion
    ty_b *= submersion

    # Clamp to avoid excessive torques
    tx_b = clamp(tx_b, -MAX_DRAG_TORQUE, MAX_DRAG_TORQUE)
    ty_b = clamp(ty_b, -MAX_DRAG_TORQUE, MAX_DRAG_TORQUE)

    # Convert BODY torque to WORLD and apply
    t_world = p.rotateVector(base_quat, (tx_b, ty_b, tz_b))
    p.applyExternalTorque(body_id, -1, t_world, p.WORLD_FRAME)

def apply_hydrodynamic_forces(body_id, base_pos, base_quat, lin_world, ang_world):
    """
    Fossen hydrodynamic model (Ref: Fossen 2011, UUV Simulator HMFossen).

    Computes and applies ALL hydrodynamic forces & torques in body frame:

      1. D(ν)·ν — Linear + quadratic damping (translational & rotational)
      2. M_A·ν̇  — Added-mass reaction forces (finite-difference acceleration)
      3. C_A(ν)·ν — Added-mass Coriolis/centripetal coupling (NEW)
      4. I_A·α  — Added rotational inertia reaction torques

    All forces/torques are computed in the BODY frame, then rotated to WORLD
    for application via PyBullet. Forces applied at COM (no spurious torques).

    Returns: (f_world, t_world, saturation_flags)
    """
    if not p.isConnected():
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (False, False, False)

    # ── Transform velocities to BODY frame ──
    # Relative flow velocity (account for water current)
    vrel_world = (
        lin_world[0] - WATER_CURRENT_WORLD[0],
        lin_world[1] - WATER_CURRENT_WORLD[1],
        lin_world[2] - WATER_CURRENT_WORLD[2],
    )
    inv_q = p.invertTransform([0, 0, 0], base_quat)[1]
    vrel_body = p.rotateVector(inv_q, vrel_world)
    w_body = p.rotateVector(inv_q, ang_world)

    # Clamp velocities for stability
    u = clamp(vrel_body[0], -MAX_SPEED, MAX_SPEED)
    v = clamp(vrel_body[1], -MAX_SPEED, MAX_SPEED)
    w = clamp(vrel_body[2], -MAX_SPEED, MAX_SPEED)
    pp = clamp(w_body[0], -MAX_OMEGA, MAX_OMEGA)  # roll rate
    q = clamp(w_body[1], -MAX_OMEGA, MAX_OMEGA)   # pitch rate
    r = clamp(w_body[2], -MAX_OMEGA, MAX_OMEGA)    # yaw rate

    sat_speed = (abs(vrel_body[0]) > MAX_SPEED or abs(vrel_body[1]) > MAX_SPEED
                 or abs(vrel_body[2]) > MAX_SPEED)

    # ═══════════════════════════════════════════════════════════════════
    # 1. D(ν)·ν — DAMPING (linear + quadratic, translational + rotational)
    # ═══════════════════════════════════════════════════════════════════
    # Translational: F_drag = -D_lin·v - ½ρ·Cd·A·|v|·v
    fx_b = -LIN_DRAG_BODY[0]*u - 0.5*RHO*CD[0]*AREA[0]*abs(u)*u
    fy_b = -LIN_DRAG_BODY[1]*v - 0.5*RHO*CD[1]*AREA[1]*abs(v)*v
    fz_b = -LIN_DRAG_BODY[2]*w - 0.5*RHO*CD[2]*AREA[2]*abs(w)*w

    # Rotational: T_drag = -D_lin_ang·ω - D_quad_ang·|ω|·ω
    tx_b = -LIN_DRAG_ANG[0]*pp - QUAD_DRAG_ANG[0]*abs(pp)*pp
    ty_b = -LIN_DRAG_ANG[1]*q  - QUAD_DRAG_ANG[1]*abs(q)*q
    tz_b = -LIN_DRAG_ANG[2]*r  - QUAD_DRAG_ANG[2]*abs(r)*r

    # ═══════════════════════════════════════════════════════════════════
    # 2. M_A·ν̇ — ADDED-MASS REACTION (translational)
    # ═══════════════════════════════════════════════════════════════════
    # Estimate body-frame acceleration via finite difference + low-pass filter.
    # F_added = -M_A · a_rel (opposes acceleration of vehicle relative to fluid)
    global LAST_VREL_BODY, LAST_A_BODY

    if LAST_VREL_BODY is None:
        LAST_VREL_BODY = (u, v, w)
        LAST_A_BODY = (0.0, 0.0, 0.0)

    try:
        ax_raw = (u - LAST_VREL_BODY[0]) / DT
        ay_raw = (v - LAST_VREL_BODY[1]) / DT
        az_raw = (w - LAST_VREL_BODY[2]) / DT
    except (TypeError, ZeroDivisionError):
        ax_raw = ay_raw = az_raw = 0.0

    MAX_ACCEL = 50.0  # m/s² clamp
    ax_raw = clamp(ax_raw, -MAX_ACCEL, MAX_ACCEL)
    ay_raw = clamp(ay_raw, -MAX_ACCEL, MAX_ACCEL)
    az_raw = clamp(az_raw, -MAX_ACCEL, MAX_ACCEL)

    alpha = ACCEL_FILTER_ALPHA
    ax = alpha * ax_raw + (1.0 - alpha) * LAST_A_BODY[0]
    ay = alpha * ay_raw + (1.0 - alpha) * LAST_A_BODY[1]
    az = alpha * az_raw + (1.0 - alpha) * LAST_A_BODY[2]

    fx_b += -ADDED_MASS_BODY[0] * ax
    fy_b += -ADDED_MASS_BODY[1] * ay
    fz_b += -ADDED_MASS_BODY[2] * az

    # ═══════════════════════════════════════════════════════════════════
    # 3. C_A(ν)·ν — ADDED-MASS CORIOLIS/CENTRIPETAL (Fossen Eq. 8.22)
    # ═══════════════════════════════════════════════════════════════════
    # For diagonal M_A = diag(Xu, Yv, Zw) and I_A = diag(Kp, Mq, Nr):
    #
    # The Coriolis force (on the vehicle, sign convention for RHS) is:
    #   F_C = -C_A(ν) · ν
    #
    # Translational Coriolis forces (body frame):
    #   Fc_x =  (Zw·w)·q - (Yv·v)·r
    #   Fc_y =  (Xu·u)·r - (Zw·w)·p
    #   Fc_z =  (Yv·v)·p - (Xu·u)·q
    #
    # Rotational Coriolis torques (body frame):
    #   Tc_x =  (Zw·w)·v - (Yv·v)·w  + (Nr·r)·q - (Mq·q)·r
    #   Tc_y =  (Xu·u)·w - (Zw·w)·u  + (Kp·p)·r - (Nr·r)·p
    #   Tc_z =  (Yv·v)·u - (Xu·u)·v  + (Mq·q)·p - (Kp·p)·q
    #
    Xu, Yv, Zw = ADDED_MASS_BODY
    Kp, Mq, Nr = ADDED_INERTIA_BODY
    sc = CORIOLIS_SCALE

    # Translational Coriolis forces
    fc_x = sc * ( (Zw*w)*q - (Yv*v)*r )
    fc_y = sc * ( (Xu*u)*r - (Zw*w)*pp )
    fc_z = sc * ( (Yv*v)*pp - (Xu*u)*q )

    fx_b += fc_x
    fy_b += fc_y
    fz_b += fc_z

    # Rotational Coriolis torques
    tc_x = sc * ( (Zw*w)*v - (Yv*v)*w + (Nr*r)*q - (Mq*q)*r )
    tc_y = sc * ( (Xu*u)*w - (Zw*w)*u + (Kp*pp)*r - (Nr*r)*pp )
    tc_z = sc * ( (Yv*v)*u - (Xu*u)*v + (Mq*q)*pp - (Kp*pp)*q )

    tx_b += tc_x
    ty_b += tc_y
    tz_b += tc_z

    # ═══════════════════════════════════════════════════════════════════
    # 4. I_A·α — ADDED ROTATIONAL INERTIA (angular acceleration reaction)
    # ═══════════════════════════════════════════════════════════════════
    global LAST_W_BODY, LAST_ALPHA_BODY
    if LAST_W_BODY is None:
        LAST_W_BODY = (pp, q, r)
        LAST_ALPHA_BODY = (0.0, 0.0, 0.0)

    try:
        alp_x_raw = (pp - LAST_W_BODY[0]) / DT
        alp_y_raw = (q  - LAST_W_BODY[1]) / DT
        alp_z_raw = (r  - LAST_W_BODY[2]) / DT
    except (TypeError, ZeroDivisionError):
        alp_x_raw = alp_y_raw = alp_z_raw = 0.0

    MAX_ANG_ACCEL = 100.0
    alp_x_raw = clamp(alp_x_raw, -MAX_ANG_ACCEL, MAX_ANG_ACCEL)
    alp_y_raw = clamp(alp_y_raw, -MAX_ANG_ACCEL, MAX_ANG_ACCEL)
    alp_z_raw = clamp(alp_z_raw, -MAX_ANG_ACCEL, MAX_ANG_ACCEL)

    alp_x = alpha * alp_x_raw + (1.0 - alpha) * LAST_ALPHA_BODY[0]
    alp_y = alpha * alp_y_raw + (1.0 - alpha) * LAST_ALPHA_BODY[1]
    alp_z = alpha * alp_z_raw + (1.0 - alpha) * LAST_ALPHA_BODY[2]

    tx_b += -ADDED_INERTIA_BODY[0] * alp_x
    ty_b += -ADDED_INERTIA_BODY[1] * alp_y
    tz_b += -ADDED_INERTIA_BODY[2] * alp_z

    # ═══════════════════════════════════════════════════════════════════
    # CLAMP & APPLY
    # ═══════════════════════════════════════════════════════════════════
    fx_b = clamp(fx_b, -MAX_DRAG_FORCE, MAX_DRAG_FORCE)
    fy_b = clamp(fy_b, -MAX_DRAG_FORCE, MAX_DRAG_FORCE)
    fz_b = clamp(fz_b, -MAX_DRAG_FORCE, MAX_DRAG_FORCE)

    tx_b = clamp(tx_b, -MAX_DRAG_TORQUE, MAX_DRAG_TORQUE)
    ty_b = clamp(ty_b, -MAX_DRAG_TORQUE, MAX_DRAG_TORQUE)
    tz_b = clamp(tz_b, -MAX_DRAG_TORQUE, MAX_DRAG_TORQUE)

    # Rotate to WORLD frame and apply
    f_world = p.rotateVector(base_quat, (fx_b, fy_b, fz_b))
    t_world = p.rotateVector(base_quat, (tx_b, ty_b, tz_b))

    p.applyExternalForce(body_id, -1, f_world, base_pos, p.WORLD_FRAME)
    p.applyExternalTorque(body_id, -1, t_world, p.WORLD_FRAME)

    # ── Update stored state for next-step finite difference ──
    try:
        LAST_VREL_BODY = (u, v, w)
        LAST_A_BODY = (ax, ay, az)
        LAST_W_BODY = (pp, q, r)
        LAST_ALPHA_BODY = (alp_x, alp_y, alp_z)
    except (NameError, TypeError):
        LAST_VREL_BODY = (0.0, 0.0, 0.0)
        LAST_A_BODY = (0.0, 0.0, 0.0)
        LAST_W_BODY = (0.0, 0.0, 0.0)
        LAST_ALPHA_BODY = (0.0, 0.0, 0.0)

    return f_world, t_world, (False, False, sat_speed)

# Keep old name as alias for backward compatibility (tests, etc.)
apply_drag = apply_hydrodynamic_forces


# ============================================
# ASSIST MODE CONTROLLERS
# ============================================

def apply_depth_hold(body_id, base_pos, lin_world):
    """
    PD depth-hold controller.  Applies a vertical force to maintain the
    target depth captured when depth hold was engaged.

    Returns the applied force magnitude for telemetry.
    """
    global DEPTH_HOLD_TARGET
    if not DEPTH_HOLD_ENABLED or DEPTH_HOLD_TARGET is None:
        return 0.0
    if not p.isConnected():
        return 0.0

    current_depth = max(0.0, SURFACE_Z - base_pos[2])
    depth_error = DEPTH_HOLD_TARGET - current_depth  # positive = too shallow, need to go down
    vz = lin_world[2]  # vertical velocity (positive = up)

    # PD: force = Kp * error - Kd * vz
    # error > 0 means we're above target depth → need downward force (negative z)
    # So: f_z = -Kp * error - Kd * vz
    fz = -DEPTH_HOLD_KP * depth_error - DEPTH_HOLD_KD * vz
    fz = clamp(fz, -DEPTH_HOLD_MAX_FORCE, DEPTH_HOLD_MAX_FORCE)

    p.applyExternalForce(body_id, -1, (0.0, 0.0, fz), base_pos, p.WORLD_FRAME)
    return fz


def apply_heading_hold(body_id, base_quat, ang_world):
    """
    PD heading-hold controller.  Applies a yaw torque to maintain the
    target heading captured when heading hold was engaged.

    Returns the applied torque magnitude for telemetry.
    """
    global HEADING_HOLD_TARGET
    if not HEADING_HOLD_ENABLED or HEADING_HOLD_TARGET is None:
        return 0.0
    if not p.isConnected():
        return 0.0

    _, _, yaw_current = p.getEulerFromQuaternion(base_quat)
    # Heading error (wrap to -π..π)
    heading_error = HEADING_HOLD_TARGET - yaw_current
    while heading_error > math.pi:
        heading_error -= 2.0 * math.pi
    while heading_error < -math.pi:
        heading_error += 2.0 * math.pi

    # Body-frame yaw rate
    inv_q = p.invertTransform([0, 0, 0], base_quat)[1]
    w_body = p.rotateVector(inv_q, ang_world)
    yaw_rate = w_body[2]

    # PD torque in body frame (z-axis = yaw)
    tz_b = HEADING_HOLD_KP * heading_error - HEADING_HOLD_KD * yaw_rate
    tz_b = clamp(tz_b, -HEADING_HOLD_MAX_TORQUE, HEADING_HOLD_MAX_TORQUE)

    # Rotate to world frame and apply
    t_world = p.rotateVector(base_quat, (0.0, 0.0, tz_b))
    p.applyExternalTorque(body_id, -1, t_world, p.WORLD_FRAME)
    return tz_b


def apply_environment_preset(preset_key):
    """
    Apply an environment preset by updating the relevant global constants.
    Call before PyBullet world setup.
    """
    global SURFACE_Z, SEABED_Z, WATER_CURRENT_BASE, WATER_CURRENT_WORLD
    global CURRENT_VARIATION_AMP, CURRENT_VARIATION_PERIOD
    global WATER_FOG_DEPTH_RANGE, WATER_FOG_COLOR_SURFACE, WATER_FOG_COLOR_DEEP
    global NUM_OBSTACLES, ACTIVE_ENVIRONMENT

    if preset_key not in ENVIRONMENT_PRESETS:
        print(f"[ENV] Unknown preset '{preset_key}', keeping current settings")
        return

    preset = ENVIRONMENT_PRESETS[preset_key]
    SURFACE_Z = preset["surface_z"]
    SEABED_Z = preset["seabed_z"]
    WATER_CURRENT_BASE = preset["current_base"]
    WATER_CURRENT_WORLD = list(WATER_CURRENT_BASE)
    CURRENT_VARIATION_AMP = preset["current_var_amp"]
    CURRENT_VARIATION_PERIOD = preset["current_var_period"]
    WATER_FOG_DEPTH_RANGE = preset["fog_depth_range"]
    WATER_FOG_COLOR_SURFACE = preset["fog_color_surface"]
    WATER_FOG_COLOR_DEEP = preset["fog_color_deep"]
    NUM_OBSTACLES = preset["num_obstacles"]
    ACTIVE_ENVIRONMENT = preset_key
    print(f"[ENV] Preset applied: {preset['label']} — {preset['description']}")


def create_environment(preset_key=None):
    """
    Build the environment (pool, risers, depth markers, etc.) in PyBullet
    based on the active or specified preset.  Returns a list of body IDs.
    """
    if preset_key is not None:
        apply_environment_preset(preset_key)
    preset = ENVIRONMENT_PRESETS.get(ACTIVE_ENVIRONMENT, ENVIRONMENT_PRESETS["pool"])

    pool_half_x = preset["pool_half_x"]
    pool_half_y = preset["pool_half_y"]
    pool_depth = SURFACE_Z - SEABED_Z

    env_ids = []

    # Seabed collision plane
    p.loadURDF("plane.urdf", [0, 0, SEABED_Z])

    # Water surface visual indicator
    try:
        surface_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[pool_half_x + 2, pool_half_y + 2, 0.002],
                                           rgbaColor=[0.3, 0.6, 0.8, 0.1])
        env_ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=surface_vis,
                          basePosition=[0, 0, SURFACE_Z]))
    except pybullet.error:
        pass

    WALL_THICK = 0.02
    WALL_COLOR = [0.55, 0.75, 0.85, 0.15]
    WALL_COLOR_FLOOR = [0.3, 0.28, 0.22, 0.7]

    try:
        # Pool floor
        floor_vis = p.createVisualShape(p.GEOM_BOX,
            halfExtents=[pool_half_x, pool_half_y, 0.02],
            rgbaColor=WALL_COLOR_FLOOR)
        env_ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=floor_vis,
            basePosition=[0, 0, SEABED_Z + 0.02]))

        # Grid lines
        grid_step_x = max(1, int(pool_half_x))
        grid_step_y = max(1, int(pool_half_y))
        for gx in range(-grid_step_x, grid_step_x + 1):
            env_ids.append(p.addUserDebugLine(
                [gx, -pool_half_y, SEABED_Z + 0.03],
                [gx, pool_half_y, SEABED_Z + 0.03],
                [0.4, 0.38, 0.32], 1, lifeTime=0))
        for gy in range(-grid_step_y, grid_step_y + 1):
            env_ids.append(p.addUserDebugLine(
                [-pool_half_x, gy, SEABED_Z + 0.03],
                [pool_half_x, gy, SEABED_Z + 0.03],
                [0.4, 0.38, 0.32], 1, lifeTime=0))

        # Four walls
        wall_vis = p.createVisualShape(p.GEOM_BOX,
            halfExtents=[WALL_THICK, pool_half_y, pool_depth / 2],
            rgbaColor=WALL_COLOR)
        env_ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=wall_vis,
            basePosition=[pool_half_x, 0, SEABED_Z + pool_depth / 2]))
        env_ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=wall_vis,
            basePosition=[-pool_half_x, 0, SEABED_Z + pool_depth / 2]))
        wall_vis_side = p.createVisualShape(p.GEOM_BOX,
            halfExtents=[pool_half_x, WALL_THICK, pool_depth / 2],
            rgbaColor=WALL_COLOR)
        env_ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=wall_vis_side,
            basePosition=[0, pool_half_y, SEABED_Z + pool_depth / 2]))
        env_ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=wall_vis_side,
            basePosition=[0, -pool_half_y, SEABED_Z + pool_depth / 2]))

        # Risers
        num_risers = preset.get("num_risers", 0)
        RISER_RADIUS = 0.08
        RISER_HEIGHT = pool_depth + 0.3
        RISER_COLOR = [0.45, 0.42, 0.40, 0.9]
        RISER_STRIPE = [0.9, 0.6, 0.1, 0.85]

        # Generate riser positions spread across the environment
        riser_positions = []
        if num_risers >= 1:
            riser_positions.append((min(1.5, pool_half_x * 0.5), 0.5, SEABED_Z + RISER_HEIGHT / 2))
        if num_risers >= 2:
            riser_positions.append((min(1.5, pool_half_x * 0.5), -0.5, SEABED_Z + RISER_HEIGHT / 2))
        if num_risers >= 3:
            riser_positions.append((min(2.0, pool_half_x * 0.6), 0.0, SEABED_Z + RISER_HEIGHT / 2))
        for ri in range(3, num_risers):
            rx = min(pool_half_x * 0.7, 1.0 + ri * 0.8)
            ry = ((-1) ** ri) * (0.3 + ri * 0.2)
            ry = clamp(ry, -pool_half_y * 0.8, pool_half_y * 0.8)
            riser_positions.append((rx, ry, SEABED_Z + RISER_HEIGHT / 2))

        for rx, ry, rz in riser_positions:
            riser_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=RISER_RADIUS, height=RISER_HEIGHT)
            riser_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=RISER_RADIUS, length=RISER_HEIGHT,
                                             rgbaColor=RISER_COLOR)
            rid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=riser_col,
                                     baseVisualShapeIndex=riser_vis,
                                     basePosition=[rx, ry, rz])
            env_ids.append(rid)

            # Inspection stripe
            stripe_vis = p.createVisualShape(p.GEOM_CYLINDER,
                radius=RISER_RADIUS + 0.005, length=0.05,
                rgbaColor=RISER_STRIPE)
            env_ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=stripe_vis,
                basePosition=[rx, ry, 0.0]))

            # Base flange
            flange_vis = p.createVisualShape(p.GEOM_CYLINDER,
                radius=RISER_RADIUS * 1.5, length=0.04,
                rgbaColor=[0.35, 0.33, 0.30, 0.9])
            env_ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=flange_vis,
                basePosition=[rx, ry, SEABED_Z + 0.04]))

        # Cross-braces between first two risers (if they exist)
        if len(riser_positions) >= 2:
            r0 = riser_positions[0]
            r1 = riser_positions[1]
            brace_dy = r0[1] - r1[1]
            brace_dx = r0[0] - r1[0]
            brace_len = math.sqrt(brace_dx ** 2 + brace_dy ** 2)
            if brace_len > 0.1:
                brace_vis = p.createVisualShape(p.GEOM_CYLINDER,
                    radius=0.025, length=brace_len,
                    rgbaColor=[0.5, 0.48, 0.44, 0.8])
                brace_angle = math.atan2(brace_dy, brace_dx)
                brace_quat = p.getQuaternionFromEuler([math.pi / 2, 0, brace_angle])
                mid_x = (r0[0] + r1[0]) / 2
                mid_y = (r0[1] + r1[1]) / 2
                env_ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=brace_vis,
                    basePosition=[mid_x, mid_y, -0.3],
                    baseOrientation=brace_quat))

        # Depth markers on +Y wall
        for depth_m in range(0, int(pool_depth) + 1):
            z_mark = SURFACE_Z - depth_m
            if z_mark < SEABED_Z:
                break
            mark_vis = p.createVisualShape(p.GEOM_BOX,
                halfExtents=[0.08, 0.002, 0.01],
                rgbaColor=[1.0, 1.0, 0.2, 0.9] if depth_m % 2 == 0 else [1.0, 0.3, 0.1, 0.9])
            env_ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=mark_vis,
                basePosition=[0, pool_half_y - 0.01, z_mark]))

        print(f"[ENV] Environment: {preset['label']}")
        print(f"[ENV]   {pool_half_x * 2:.0f}m x {pool_half_y * 2:.0f}m x {pool_depth:.1f}m deep")
        print(f"[ENV]   {len(riser_positions)} risers, current base={WATER_CURRENT_BASE}")
    except pybullet.error as env_err:
        print(f"[ENV] Warning: environment setup error: {env_err}")

    return env_ids


# --- Obstacles: neutral buoyancy + drag in water ---
def apply_obstacle_water_forces(obstacle_ids):
    """Apply neutral buoyancy + drag to obstacles so they behave like objects in water."""
    if not p.isConnected() or not obstacle_ids:
        return
    for oid in obstacle_ids:
        try:
            pos, quat = p.getBasePositionAndOrientation(oid)
            lin, ang = p.getBaseVelocity(oid)
        except pybullet.error:
            continue

        # Only apply buoyancy if obstacle is below water surface
        obs_depth = max(0.0, SURFACE_Z - pos[2])
        # Smooth submersion factor for obstacles near surface
        _obs_half = 0.10  # approximate half-extent of obstacles
        if obs_depth >= _obs_half:
            obs_sub = 1.0
        elif obs_depth <= 0.0:
            obs_sub = 0.0
        else:
            obs_sub = obs_depth / _obs_half
        if obs_sub > 0.001:
            buoy = OBSTACLE_MASS * GRAVITY * OBSTACLE_BUOYANCY_SCALE * obs_sub
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

        # Angular drag — dampen spinning in water
        wx, wy, wz = ang
        w_mag = math.sqrt(wx*wx + wy*wy + wz*wz)
        if w_mag > 1e-6:
            ang_drag_coef = 0.5  # Nm per rad/s
            t_mag = ang_drag_coef * w_mag
            t_mag = clamp(t_mag, 0.0, 5.0)
            p.applyExternalTorque(oid, -1, (-t_mag * wx / w_mag, -t_mag * wy / w_mag, -t_mag * wz / w_mag), p.WORLD_FRAME)

        # Soft depth-hold restoring force: gently push obstacles back toward
        # their spawn z-height.  This prevents slow drift from numerical errors
        # in the buoyancy==gravity balance.  The spring is very soft (2 N/m)
        # so collisions / ROV thrust can still move them easily.
        target_z = OBSTACLE_SPAWN_Z.get(oid, pos[2])
        dz_err = target_z - pos[2]
        depth_hold_k = 2.0   # N per m of deviation
        depth_hold_d = 1.0   # N per m/s (damping)
        fz_hold = depth_hold_k * dz_err - depth_hold_d * lin[2]
        fz_hold = clamp(fz_hold, -3.0, 3.0)
        p.applyExternalForce(oid, -1, (0.0, 0.0, fz_hold), pos, p.WORLD_FRAME)


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
    # Do NOT pass rgbaColor so PyBullet uses the per-face materials from the MTL file.
    try:
        vis = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=OBJ_FILE,
            meshScale=MESH_SCALE,
            visualFramePosition=(-center[0], -center[1], -center[2]),
            visualFrameOrientation=(0, 0, 0, 1),
        )
    except pybullet.error:
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
    except pybullet.error:
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
# Maps obstacle body ID → spawn z-position (for depth-hold restoring force)
OBSTACLE_SPAWN_Z = {}

def spawn_obstacles(n=NUM_OBSTACLES):
    """Spawn simple dynamic boxes/spheres you can drag with the mouse or move via keys."""
    global OBSTACLE_SPAWN_Z
    ids = []
    OBSTACLE_SPAWN_Z = {}
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
        p.changeDynamics(bid, -1, linearDamping=0.0, angularDamping=0.0, lateralFriction=0.7, restitution=0.05)
        OBSTACLE_SPAWN_Z[bid] = pz
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
    """Legacy arrow update — kept as fallback but not used by default."""
    if not p.isConnected():
        return arrow_id
    rel = p.rotateVector(base_quat, thr["pos"])
    p_world = (base_pos[0] + rel[0], base_pos[1] + rel[1], base_pos[2] + rel[2])
    d_world = p.rotateVector(base_quat, thr["dir"])
    tip = (p_world[0] + d_world[0] * 0.35, p_world[1] + d_world[1] * 0.35, p_world[2] + d_world[2] * 0.35)
    mag = abs(level)
    if mag < 1e-3:
        return p.addUserDebugLine(p_world, p_world, [0.2, 0.2, 0.2], 1, lifeTime=0, replaceItemUniqueId=arrow_id)
    if level < 0:
        tip = (p_world[0] - d_world[0] * 0.35, p_world[1] - d_world[1] * 0.35, p_world[2] - d_world[2] * 0.35)
        color = [0.2, 0.6, 1.0]
    else:
        color = [1.0, 0.65, 0.1]
    width = max(1, min(8, int(2 + mag * 6)))
    return p.addUserDebugLine(p_world, tip, color, width, lifeTime=0, replaceItemUniqueId=arrow_id)


# ============================================
# THRUSTER INDICATORS (free-standing cone bodies, repositioned each frame)
# ============================================
# Each thruster gets a small cylinder visual (mass=0, no collision) that is
# manually positioned via resetBasePositionAndOrientation every frame.
# No constraints are used — constraint reaction forces fight the thrusters.
# Color is updated via changeVisualShape only on state change (rare).

THR_IND_CONE_RADIUS = 0.018  # cone base radius (m)
THR_IND_CONE_LENGTH_MIN = 0.04    # length when just starting to ramp up
THR_IND_CONE_LENGTH_MAX = 0.14    # length at full thrust
THR_IND_COLOR_OFF  = [0.35, 0.35, 0.35, 0.15]  # grey, nearly invisible when off
THR_IND_COLOR_FWD  = [1.0, 0.55, 0.05, 0.9]    # bright orange for forward thrust
THR_IND_COLOR_REV  = [0.15, 0.5, 1.0, 0.9]     # blue for reverse thrust
THR_IND_COLOR_WARN = [1.0, 0.1, 0.1, 1.0]      # red flash for proximity warning

def _thr_dir_to_quat(d):
    """Compute quaternion that rotates +Z (PyBullet cone axis) to direction d."""
    dx, dy, dz = d
    # Target direction (unit)
    mag = math.sqrt(dx*dx + dy*dy + dz*dz)
    if mag < 1e-9:
        return (0, 0, 0, 1)
    dx, dy, dz = dx/mag, dy/mag, dz/mag
    # +Z axis
    ux, uy, uz = 0.0, 0.0, 1.0
    # cross product (rotation axis)
    cx = uy*dz - uz*dy
    cy = uz*dx - ux*dz
    cz = ux*dy - uy*dx
    cmag = math.sqrt(cx*cx + cy*cy + cz*cz)
    dot = ux*dx + uy*dy + uz*dz
    if cmag < 1e-9:
        # Parallel or anti-parallel
        if dot > 0:
            return (0, 0, 0, 1)
        else:
            # 180° rotation about X
            return (1, 0, 0, 0)
    # half-angle quaternion: q = (sin(θ/2)*axis, cos(θ/2))
    # angle = acos(dot), but use atan2 for stability
    angle = math.atan2(cmag, dot)
    ha = angle / 2.0
    s = math.sin(ha) / cmag
    return (cx*s, cy*s, cz*s, math.cos(ha))


def create_thruster_indicators(rov_id, thrusters):
    """
    Create small cylinder visual bodies for each thruster.

    These are FREE-STANDING bodies (mass=0, no collision, no constraints).
    We manually position them each frame using resetBasePositionAndOrientation,
    which is 1 cheap IPC call per thruster.  Color is only changed when
    thruster state changes (changeVisualShape — rare).

    IMPORTANT: No constraints are used because PyBullet's constraint solver
    applies reaction forces to the parent body, which fights the thrusters
    and pins the ROV in place.
    """
    # Get initial ROV pose so indicators start at the correct position
    # instead of appearing at the world origin for one frame.
    try:
        init_pos, init_quat = p.getBasePositionAndOrientation(rov_id)
    except pybullet.error:
        init_pos, init_quat = (0, 0, 0), (0, 0, 0, 1)

    indicators = []
    for t in thrusters:
        dx, dy, dz = t["dir"]
        quat = _thr_dir_to_quat((dx, dy, dz))

        # Pre-compute the body-frame offset (center of cylinder along thrust dir)
        half_len = THR_IND_CONE_LENGTH_MAX / 2.0
        offset_body = (
            t["pos"][0] + dx * half_len,
            t["pos"][1] + dy * half_len,
            t["pos"][2] + dz * half_len,
        )

        # Compute initial world-frame position & orientation
        off_w = p.rotateVector(init_quat, offset_body)
        init_world_pos = (init_pos[0] + off_w[0],
                          init_pos[1] + off_w[1],
                          init_pos[2] + off_w[2])
        init_world_quat = p.multiplyTransforms(
            [0, 0, 0], init_quat,
            [0, 0, 0], quat
        )[1]

        # Visual: small cylinder pointing along thrust direction
        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=THR_IND_CONE_RADIUS,
            length=THR_IND_CONE_LENGTH_MAX,
            rgbaColor=THR_IND_COLOR_OFF,
        )

        # No collision — visual only, mass 0 = static (won't fall)
        body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=list(init_world_pos),
            baseOrientation=list(init_world_quat),
        )

        indicators.append({
            "body": body,
            "state": 0,
            "offset_body": offset_body,
            "quat_body": quat,  # orientation in body frame
            "thr_dir": (dx, dy, dz),
            "thr_pos": t["pos"],
        })

    return indicators


def update_thruster_indicators(indicators, base_pos, base_quat, thr_levels, proximity_warn=False):
    """
    Reposition all thruster indicators to follow the ROV AND update colors
    on state change.  The indicator body-frame offset is dynamically adjusted
    based on thr_level magnitude so the cone "grows" with thrust ramp-up.

    Cost: 4× resetBasePositionAndOrientation per frame
    (very cheap — no debug line recreation) + occasional changeVisualShape.
    """
    # Blink phase for proximity warning (computed once, not per-thruster)
    _blink_on = False
    if proximity_warn:
        _blink_on = int(time.monotonic() * 4) % 2 == 0

    for i, ind in enumerate(indicators):
        level = thr_levels[i] if i < len(thr_levels) else 0.0
        abs_level = abs(level)

        # Dynamic offset: scale cone position along thrust dir by ramp level
        # so the indicator appears to extend as thrust builds
        dx, dy, dz = ind["thr_dir"]
        lerp = abs_level  # 0..1
        cur_half_len = (THR_IND_CONE_LENGTH_MIN + lerp * (THR_IND_CONE_LENGTH_MAX - THR_IND_CONE_LENGTH_MIN)) / 2.0
        offset = (
            ind["thr_pos"][0] + dx * cur_half_len,
            ind["thr_pos"][1] + dy * cur_half_len,
            ind["thr_pos"][2] + dz * cur_half_len,
        )

        # Position: transform body-frame offset to world
        off_w = p.rotateVector(base_quat, offset)
        pos_w = (base_pos[0] + off_w[0],
                 base_pos[1] + off_w[1],
                 base_pos[2] + off_w[2])

        # Orientation: body rotation × local cylinder rotation
        quat_w = p.multiplyTransforms(
            [0, 0, 0], base_quat,
            [0, 0, 0], ind["quat_body"]
        )[1]

        p.resetBasePositionAndOrientation(ind["body"], pos_w, quat_w)

        # Color change only on state transition
        # Threshold matches physics zero-flush to prevent stuck indicator colors
        if abs_level <= 1e-4:
            new_state = 0
        elif level > 1e-6:  # Small hysteresis
            new_state = 1
        else:
            new_state = -1

        # Override: flash red on proximity warning (toggle every ~0.25s)
        if proximity_warn and new_state == 0:
            # Blink idle indicators red as a warning
            new_state = 99 if _blink_on else 0

        if new_state != ind["state"]:
            ind["state"] = new_state
            if new_state == 0:
                color = THR_IND_COLOR_OFF
            elif new_state == 1:
                color = THR_IND_COLOR_FWD
            elif new_state == 99:
                color = THR_IND_COLOR_WARN
            else:
                color = THR_IND_COLOR_REV
            try:
                p.changeVisualShape(ind["body"], -1, rgbaColor=color)
            except pybullet.error:
                pass


def setup_pybullet():
    """Initialize PyBullet physics server and configure rendering options."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()

    # Disable expensive rendering features for performance
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
    except pybullet.error:
        pass

    p.setGravity(0, 0, -GRAVITY)
    p.setPhysicsEngineParameter(fixedTimeStep=DT, numSolverIterations=50, numSubSteps=1)


def teardown_simulation(_rec_active, _rec_writer, _rec_frame_count, _rec_path):
    """Clean up resources on simulator exit."""
    global _log_file_handle

    # Finalize any active recording
    if _rec_active and _rec_writer is not None:
        try:
            _rec_writer.release()
            _dur = _rec_frame_count / max(1, REC_FPS)
            print(f"[REC] ⏹  Recording saved on exit: {_rec_path}")
            print(f"[REC]    {_rec_frame_count} frames, ~{_dur:.1f}s duration")
        except OSError:
            pass

    if HAS_JOYSTICK and getattr(joystick_panel, "_shared", None) is not None:
        try:
            with joystick_panel._shared.get_lock():
                joystick_panel._shared[REC_STATUS] = REC_STATUS_OK
                joystick_panel._shared[REC_FLAG] = 0.0
        except (IndexError, ValueError, OSError):
            pass

    try:
        p.disconnect()
    except pybullet.error:
        pass
    if ENABLE_JOYSTICK_PANEL and HAS_JOYSTICK:
        try:
            joystick_panel.stop_joystick_panel()
        except (OSError, RuntimeError):
            pass

    # Close log file
    if _log_file_handle is not None:
        try:
            _log_file_handle.write("\n" + "=" * 80 + "\n")
            _log_file_handle.write(f"Simulation ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            _log_file_handle.close()
            print(f"✅ Log saved to: {LOG_FILE}")
        except OSError:
            pass
        _log_file_handle = None


def main():
    global _log_file_handle, USER_QUIT, THRUST_LEVEL, WATER_CURRENT_WORLD
    global CAM_CHASE_ENABLED, TRAIL_ENABLED, AUTOTEST, AUTOTEST_EXIT
    global LAST_VREL_BODY, LAST_A_BODY, LAST_W_BODY, LAST_ALPHA_BODY
    global OBJ_FILE, GLTF_FILE, ACTIVE_CONFIG_NAME
    global DEPTH_HOLD_ENABLED, DEPTH_HOLD_TARGET, HEADING_HOLD_ENABLED, HEADING_HOLD_TARGET
    global PROPORTIONAL_MODE, THRUSTER_FAILURE_ENABLED, THRUSTER_FAILED
    global SHOW_FORCE_VECTORS, ACTIVE_ENVIRONMENT, NUM_OBSTACLES
    
    # Initialize log file
    try:
        _log_file_handle = open(LOG_FILE, 'w')
        _log_file_handle.write(f"ROV Simulator Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        _log_file_handle.write("=" * 80 + "\n\n")
        # Write a small CSV header for structured physics logging (easy to import)
        if LOG_PHYSICS_DETAILED and _log_file_handle is not None:
            _log_file_handle.write("# DETAILED_PHYSICS_CSV\n")
            obs_hdrs = ""
            for oi in range(NUM_OBSTACLES):
                obs_hdrs += f",obs{oi}_x,obs{oi}_y,obs{oi}_z,obs{oi}_vx,obs{oi}_vy,obs{oi}_vz"
            hdr = (
                "time,step,px,py,pz, vx,vy,vz, vbx,vby,vbz, wx,wy,wz, wbx,wby,wbz,"
                "Fthr_x,Fthr_y,Fthr_z, Tthr_x,Tthr_y,Tthr_z, Fdrag_x,Fdrag_y,Fdrag_z, Tdrag_x,Tdrag_y,Tdrag_z,"
                "buoy_z,ballast_z, depth, buoy_factor,"
                "current_x,current_y,current_z, roll_deg,pitch_deg,yaw_deg, thr_levels"
                + obs_hdrs + "\n"
            )
            _log_file_handle.write(hdr)
        print(f"📝 Logging to: {LOG_FILE}")
    except OSError as e:
        print(f"⚠️  Could not open log file: {e}")

    # ── Thruster configuration selector (before PyBullet GUI opens) ──
    _autotest_mode = os.environ.get("ROV_AUTOTEST", "0") == "1"
    if not _autotest_mode and len(THRUSTER_CONFIGS) > 1:
        try:
            sel = choose_thruster_config()
            if sel is None:
                print("[CONFIG] No configuration selected — exiting.")
                return
        except (RuntimeError, ImportError) as e:
            print(f"⚠️  Tkinter selector crashed: {e}")
            print("[CONFIG] Using auto-fallback to first config.")
            if THRUSTER_CONFIGS:
                name = list(THRUSTER_CONFIGS.keys())[0]
                OBJ_FILE  = THRUSTER_CONFIGS[name]["obj"]
                GLTF_FILE = THRUSTER_CONFIGS[name]["gltf"]
                ACTIVE_CONFIG_NAME = name
                print(f"[CONFIG] Auto-selected: {name}")
    else:
        if THRUSTER_CONFIGS:
            name = list(THRUSTER_CONFIGS.keys())[0]
            OBJ_FILE  = THRUSTER_CONFIGS[name]["obj"]
            GLTF_FILE = THRUSTER_CONFIGS[name]["gltf"]
            ACTIVE_CONFIG_NAME = name
            print(f"[CONFIG] Auto-selected: {name}")

    setup_pybullet()

    # ── Build environment from preset ──
    _env_ids = create_environment(ACTIVE_ENVIRONMENT)

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
            # Thruster directions from GLTF are negated for horizontal thrusters
            # so that cmd=+1 means "push ROV forward" (toward camera-forward, mesh -Y).
            # Vertical thrusters are normalised to point +Z (up).
            # DDR: rear thrusters (1,2) angled ~45° outward, front thruster (4) straight,
            # vertical thruster (3) points up.  T1/T2 create yaw torque differentially.
            print("[AUTO] Thrusters detected from GLTF:")
            for t in THRUSTERS:
                print("   ", t)
        else:
            print("[AUTO] GLTF thruster detect failed; using fallback THRUSTERS.")

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
        print(f"[CAM] Using camera pose from GLTF: pos={tuple(round(x,3) for x in cam_pos_body)}, "
              f"fwd={tuple(round(x,3) for x in cam_fwd_body)}, up={tuple(round(x,3) for x in cam_up_body)}")
        print(f"[CAM] Servo tilt range: {CAMERA_SERVO_MIN_DEG:.0f}° .. {CAMERA_SERVO_MAX_DEG:.0f}°")

    markers = make_markers(THRUSTERS)

    # Thruster indicators: small free-standing cone bodies positioned each frame.
    thr_indicators = []
    if ENABLE_THRUSTER_ARROWS:
        thr_indicators = create_thruster_indicators(rov, THRUSTERS)
        print(f"[THR] Created {len(thr_indicators)} thruster indicators (free-standing, repositioned each frame)")

    # Legacy debug-line arrows removed — they created zero-length lines at the
    # origin that showed as visible artefacts.  Thruster indicators are the only
    # visual system now.
    arrows = []

    # HUD text items (screen overlay using addUserDebugText)
    hud_items = {}
    # Shadow text ID for dark outline behind bright HUD (improves readability)
    _hud_shadow_id = None
    if HUD_ENABLED:
        # Single combined HUD text item — 1 IPC call instead of 5 for much less lag
        # Shadow layer (dark) rendered first, then bright layer on top
        _hud_shadow_id = p.addUserDebugText(
            "Initializing...",
            [0, 0, 0],
            textColorRGB=[0.0, 0.0, 0.0], textSize=1.15, lifeTime=0)
        hud_items["combined"] = p.addUserDebugText(
            "Initializing...",
            [0, 0, 0],
            textColorRGB=[1.0, 1.0, 1.0], textSize=1.1, lifeTime=0)
    hud_every = max(1, int(round((1.0 / DT) / HUD_UPDATE_HZ)))

    thr_on = [False] * len(THRUSTERS)
    thr_reverse = [False] * len(THRUSTERS)  # Track reverse mode for each thruster
    thr_cmd = [0.0] * len(THRUSTERS)   # -1 to 1 command (neg=reverse, pos=forward)
    thr_level = [0.0] * len(THRUSTERS) # actual ramped level -1..1
    for i, m in enumerate(markers):
        set_marker(m, False)

    # Joystick panel (separate Tkinter window) — skip during AUTOTEST
    _autotest_mode = os.environ.get("ROV_AUTOTEST", "0") == "1"
    if ENABLE_JOYSTICK_PANEL and HAS_JOYSTICK and not _autotest_mode:
        joystick_panel.start_joystick_panel()
        print("[JOYSTICK] Virtual joystick panel opened")
    # Per-thruster cooldown tracking: last time each thruster changed sign
    _js_last_sign = [0] * len(THRUSTERS)      # -1, 0, +1
    _js_last_switch_t = [0.0] * len(THRUSTERS) # wall-clock time of last sign change

    # AUTOTEST: If ROV_AUTOTEST=1 in the environment, simulate key actions so
    # users without GUI focus can validate thruster toggle/reverse logic.
    AUTOTEST = os.environ.get("ROV_AUTOTEST", "0") == "1"
    AUTOTEST_EXIT = os.environ.get("ROV_AUTOTEST_EXIT", "0") == "1"  # auto-exit when done
    if AUTOTEST:
        print("[AUTOTEST] Running comprehensive test sequence (ROV_AUTOTEST=1)")
        # Comprehensive schedule: idle → each thruster → combos → vertical → all off → settle
        autotest_schedule = {
            # Phase 0: idle settling (0–2s)
            # Phase 1: T1 horizontal (2–4s)
            int(2.0 / DT):  "t1_on",
            int(3.0 / DT):  "t1_rev",
            int(4.0 / DT):  "t1_off",
            # Phase 2: T2 horizontal (4–6s)
            int(4.5 / DT):  "t2_on",
            int(5.5 / DT):  "t2_rev",
            int(6.0 / DT):  "t2_off",
            # Phase 3: T4 forward/back (6–8s)
            int(6.5 / DT):  "t4_on",
            int(7.5 / DT):  "t4_rev",
            int(8.0 / DT):  "t4_off",
            # Phase 4: T3 vertical up/down (8–10s)
            int(8.5 / DT):  "t3_on",
            int(9.5 / DT):  "t3_rev",
            int(10.0 / DT): "t3_off",
            # Phase 5: T1+T2 both forward (differential drive) (10–12s)
            int(10.5 / DT): "t1t2_on",
            int(12.0 / DT): "t1t2_off",
            # Phase 6: T1+T2 opposed (yaw test) (12–14s)
            int(12.5 / DT): "t1_on",
            int(12.5 / DT + 1): "t2_on_rev",
            int(14.0 / DT): "all_off",
            # Phase 7: All 4 on briefly (14–16s)
            int(14.5 / DT): "all_on",
            int(16.0 / DT): "all_off2",
            # Phase 8: idle settle (16–20s) — observe drift/stability
            int(20.0 / DT): "done",
        }

    # Camera (local state, no globals)
    cam_dist = CAM_DIST
    cam_yaw = CAM_YAW
    cam_pitch = CAM_PITCH
    cam_target = list(CAM_TARGET)
    cam_follow = CAM_FOLLOW
    p.resetDebugVisualizerCamera(cameraDistance=cam_dist, cameraYaw=cam_yaw, cameraPitch=cam_pitch, cameraTargetPosition=cam_target)

    sim_step = 0
    log_every = max(1, int(round((1.0 / DT) / LOG_FPS))) if LOG_FPS > 0 else 0
    log_phys_every = max(1, int(round((1.0 / DT) / LOG_PHYSICS_HZ))) if LOG_PHYSICS_HZ > 0 else 0
    vis_every = max(1, int(round((1.0 / DT) / VIS_FPS)))
    prev_every = max(1, int(round((1.0 / DT) / PREVIEW_FPS)))
    thr_vis_every = vis_every
    proj = p.computeProjectionMatrixFOV(fov=CAM_FOV, aspect=float(CAM_PREVIEW_W)/float(CAM_PREVIEW_H), nearVal=CAM_NEAR, farVal=CAM_FAR)
    last_cam_warn = False
    # Wall-clock timer for proper real-time pacing
    _wall_t0 = time.monotonic()
    _sim_clock = 0.0  # accumulated sim time
    # Periodic log file flush counter
    _flush_counter = 0

    # ── New feature state ─────────────────────────────────────────
    _proximity_warn = False               # set True when ROV is near seabed/surface
    _last_collision_log_t = 0.0           # wall-clock time of last collision message
    _emergency_surface_active = False     # True while emergency surface is engaged
    _thr_efficiency = [0.0] * len(THRUSTERS)  # actual/max ratio per thruster
    _trail_positions = []                 # breadcrumb trail [(x,y,z), ...] for path viz
    _trail_line_ids = []                  # PyBullet debug line IDs for trail segments
    _trail_last_pos = None                # last recorded trail position
    TRAIL_ENABLED = False                 # draw breadcrumb trail behind the ROV (disabled by default — IPC heavy)
    TRAIL_SPACING = 0.30                  # meters between trail points (wider = fewer IPC calls)
    TRAIL_MAX_POINTS = 150                # max trail segments (ring buffer)
    TRAIL_COLOR_RECENT = [0.0, 0.9, 1.0] # cyan
    TRAIL_COLOR_OLD    = [0.0, 0.2, 0.3] # dark teal

    # ── Timing metrics state ─────────────────────────────────────
    _timing_frame_count = 0
    _timing_physics_total = 0.0
    _timing_render_total = 0.0
    _timing_last_report = time.monotonic()
    _timing_fps_samples = []

    # ── Assist mode state ────────────────────────────────────────
    DEPTH_HOLD_ENABLED = False
    DEPTH_HOLD_TARGET = None
    HEADING_HOLD_ENABLED = False
    HEADING_HOLD_TARGET = None
    _depth_hold_force = 0.0
    _heading_hold_torque = 0.0

    # ── Thruster failure simulation state ────────────────────────
    THRUSTER_FAILED = [False] * len(THRUSTERS) if THRUSTER_FAILURE_ENABLED else []
    _thruster_fail_timer = [0.0] * len(THRUSTERS)

    # ── Debug force visualization state ──────────────────────────
    _force_viz_ids = {}  # maps label → debug line ID

    print("\n" + "🌊 "*20)
    print("╔" + "═"*68 + "╗")
    print("║" + " CTRL+SEA ROV SIMULATOR — PyBullet Physics Engine ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    print("\n" + "─"*72)
    print("📋 RUNTIME CONTROLS (TKINTER SETTINGS PANEL)")
    print("─"*72)
    print("\n🎛️ SETTINGS PANEL:")
    print("  Open from controller: SETTINGS button")
    print("  Propulsion: thrust level, proportional mode, emergency surface")
    print("  Assist: depth hold, heading hold")
    print("  Camera: follow, chase, top-down")
    print("  Diagnostics: force vectors, thruster failure, trail")
    print("  Actions: reset ROV")
    print("\n🎮 DIRECT CONTROL:")
    print("  Controller panel sticks + heave buttons drive thrusters")
    print("  Keyboard setting controls in simulator are disabled")

    if ENABLE_JOYSTICK_PANEL and HAS_JOYSTICK:
        print("\n🎮 CONTROLLER PANEL (separate window):")
        print("  Left stick — 8 angular zones (45° each):")
        print("    ↑ Fwd (T4)  ↗ Fwd+Yaw-R (T4+T1−T2)  → Yaw-R (T1−T2)")
        print("    ↘ Rev+Yaw-R (T4+T1−T2)  ↓ Rev (T4)  ↙ Rev+Yaw-L (T4+T1−T2)")
        print("    ← Yaw-L (T1−T2)  ↖ Fwd+Yaw-L (T4+T1−T2)")
        print("  Right stick: ↑↓ Camera pitch (servo tilt)  ←→ Yaw (T1−T2)")
        print("  Thrusters are ON/OFF only (100% power)")
        print(f"  Direction switch cooldown: {JOYSTICK_SWITCH_COOLDOWN:.1f}s")
    
    print("\n⚙️  PHYSICS ENGINE:")
    print(f"  Timestep: {DT*1000:.1f}ms ({1.0/DT:.0f} Hz)  │  Mass: {MASS} kg  │  Gravity: {GRAVITY} m/s²")
    print(f"  Limits: Speed {MAX_SPEED}m/s  │  Rotation {MAX_OMEGA}rad/s")
    print(f"  Thrust: Horizontal {MAX_THRUST_H}N  │  Vertical {MAX_THRUST_V}N")
    
    print("\n💧 HYDRODYNAMICS:")
    print(f"  Buoyancy: {BUOYANCY_SCALE:.2f}× weight @ {COB_OFFSET_BODY}  │  Ballast: {BALLAST_SCALE:.2f}× @ {BALLAST_OFFSET_BODY}")
    print(f"  Righting: Kp={RIGHTING_K_RP}  Kd={RIGHTING_KD_RP}")
    print(f"  Drag: ρ={RHO}kg/m³  │  Linear: {LIN_DRAG_BODY}  │  CD/Area: {CD}/{AREA}")
    print(f"  Added Mass: {ADDED_MASS_BODY} kg")
    print(f"  Ocean Current: base={WATER_CURRENT_BASE}  var amp={CURRENT_VARIATION_AMP}")
    print(f"  Surface: z={SURFACE_Z}m  │  Seabed: z={SEABED_Z}m  │  Depth buoy compress: {DEPTH_BUOYANCY_COMPRESSIBILITY}/m")
    
    print("\n🎨 RENDERING:")
    status_str = ""
    if ENABLE_CAMERA_PREVIEW:
        status_str += f"📷 Camera {PREVIEW_FPS}fps (servo tilt {CAMERA_SERVO_MIN_DEG:.0f}°..{CAMERA_SERVO_MAX_DEG:.0f}°)  "
    if ENABLE_THRUSTER_ARROWS:
        status_str += f"→ Arrows {VIS_FPS}fps  "
    if ENABLE_MARKERS:
        status_str += f"● Markers  "
    if not status_str:
        status_str = "Minimal (arrows/markers disabled)"
    print(f"  {status_str}")

    print("\n🆕 NEW FEATURES:")
    feat_list = []
    if CAMERA_OSD_ENABLED:
        feat_list.append("Camera OSD (depth bar + heading)")
    if CAMERA_DEPTH_TINT:
        feat_list.append("Depth-dependent camera tint")
    if CAMERA_SHAKE_ENABLED:
        feat_list.append("Speed-dependent camera shake")
    if PROXIMITY_WARN_ENABLED:
        feat_list.append(f"Proximity warnings (<{PROXIMITY_WARN_DIST}m)")
    if COLLISION_FEEDBACK_ENABLED:
        feat_list.append("Collision detection feedback")
    if TRAIL_ENABLED:
        feat_list.append("Breadcrumb trail (path history)")
    feat_list.append("Emergency surface via settings panel")
    feat_list.append("Chase camera via settings panel")
    feat_list.append("Top-down view via settings panel")
    feat_list.append("Dynamic thruster indicators (length scales with ramp)")
    feat_list.append("Thruster efficiency readout in telemetry")
    feat_list.append("⏺ Screen recording (REC button on controller panel)")
    feat_list.append("🤖 Depth hold + heading hold from settings panel")
    feat_list.append("🎮 Proportional joystick mode from settings panel")
    feat_list.append("📊 Force vector visualization from settings panel")
    feat_list.append("💥 Thruster failure simulation from settings panel")
    feat_list.append(f"🌊 Environment presets: {', '.join(ENVIRONMENT_PRESETS.keys())}")
    feat_list.append(f"⏱ FPS / timing metrics")
    for fi, feat in enumerate(feat_list):
        print(f"  {'├' if fi < len(feat_list)-1 else '└'}─ {feat}")
    
    print("\n" + "🌊 "*20)
    print()

    # Pre-loop variable initialisation (used by camera/OSD before first physics tick)
    depth = max(0.0, SURFACE_Z - 0.60)  # initial depth at start position z=0.60

    # ── Recording state ───────────────────────────────────────────
    _rec_active = False       # True while recording frames
    _rec_writer = None        # cv2.VideoWriter instance
    _rec_path = None          # output file path
    _rec_frame_count = 0      # frames written
    _rec_every = max(1, int(round((1.0 / DT) / REC_FPS)))  # sim steps per recorded frame
    _rec_panel_seq = 0        # last panel screenshot sequence number seen

    # ── Top-down view state ───────────────────────────────────────
    _topdown_active = False
    _topdown_saved = (cam_dist, cam_pitch, cam_yaw)  # saved camera state for restore
    _panel_last_reset_cmd = 0
    _panel_last_depth_hold = DEPTH_HOLD_ENABLED
    _panel_last_heading_hold = HEADING_HOLD_ENABLED
    _panel_last_prop_mode = PROPORTIONAL_MODE
    _panel_last_cam_follow = cam_follow
    _panel_last_cam_chase = CAM_CHASE_ENABLED
    _panel_last_topdown = _topdown_active
    _panel_last_force_viz = SHOW_FORCE_VECTORS
    _panel_last_thr_fail = THRUSTER_FAILURE_ENABLED
    _panel_last_emergency = _emergency_surface_active
    _panel_last_trail = TRAIL_ENABLED

    while p.isConnected():
        # Keyboard setting controls are intentionally disabled.
        # Runtime settings are driven from the Tkinter settings panel.
        keys = {}

        # Exit
        if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
            USER_QUIT = True
            break

        # Reset
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            # Start position: facing obstacles ahead, level orientation
            start_pos = [0, 0, 0.60]
            start_rpy = [0, 0, 0]  # Yaw 0 = facing forward (+X)
            p.resetBasePositionAndOrientation(rov, start_pos, p.getQuaternionFromEuler(start_rpy))
            p.resetBaseVelocity(rov, [0, 0, 0], [0, 0, 0])
            # Reset added-mass state to avoid phantom impulse after reset
            LAST_VREL_BODY = None
            LAST_A_BODY = (0.0, 0.0, 0.0)
            LAST_W_BODY = None
            LAST_ALPHA_BODY = (0.0, 0.0, 0.0)
            # Clear breadcrumb trail on reset
            for _tid in _trail_line_ids:
                try:
                    p.removeUserDebugItem(_tid)
                except pybullet.error:
                    pass
            _trail_positions.clear()
            _trail_line_ids.clear()
            _trail_last_pos = None
            _emergency_surface_active = False
            print(f"[SIM] Reset ROV to {start_pos} facing forward (yaw={start_rpy[2]:.0f}°)")

        # Toggle camera follow (F key)
        if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
            cam_follow = not cam_follow
            print(f"[CAM] Follow mode: {'ON' if cam_follow else 'OFF'}")

        # Toggle chase camera (G key) — auto-orbits behind ROV heading
        if ord('g') in keys and keys[ord('g')] & p.KEY_WAS_TRIGGERED:
            CAM_CHASE_ENABLED = not CAM_CHASE_ENABLED
            print(f"[CAM] Chase camera: {'ON' if CAM_CHASE_ENABLED else 'OFF'}")

        # Toggle top-down view (T key) — bird's-eye looking straight down
        if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
            _topdown_active = not _topdown_active
            if _topdown_active:
                # Save current camera state so we can restore it later
                _topdown_saved = (cam_dist, cam_pitch, cam_yaw)
                cam_pitch = -89.9   # look straight down
                cam_dist = 3.5      # zoom out to see the pool
                print("[CAM] Top-down view: ON  (press T again to restore)")
            else:
                # Restore previous camera state
                cam_dist, cam_pitch, cam_yaw = _topdown_saved
                print("[CAM] Top-down view: OFF (camera restored)")
            p.resetDebugVisualizerCamera(cameraDistance=cam_dist, cameraYaw=cam_yaw,
                                         cameraPitch=cam_pitch, cameraTargetPosition=cam_target)

        # ── Depth hold toggle (H key) ─────────────────────────────
        if ord('h') in keys and keys[ord('h')] & p.KEY_WAS_TRIGGERED:
            DEPTH_HOLD_ENABLED = not DEPTH_HOLD_ENABLED
            if DEPTH_HOLD_ENABLED:
                current_depth = max(0.0, SURFACE_Z - base_pos[2])
                DEPTH_HOLD_TARGET = current_depth
                print(f"[ASSIST] Depth hold: ON — holding {current_depth:.2f}m")
            else:
                DEPTH_HOLD_TARGET = None
                print("[ASSIST] Depth hold: OFF")

        # ── Heading hold toggle (Y key) ───────────────────────────
        if ord('y') in keys and keys[ord('y')] & p.KEY_WAS_TRIGGERED:
            HEADING_HOLD_ENABLED = not HEADING_HOLD_ENABLED
            if HEADING_HOLD_ENABLED:
                _, _, yaw_hold = p.getEulerFromQuaternion(base_quat)
                HEADING_HOLD_TARGET = yaw_hold
                print(f"[ASSIST] Heading hold: ON — holding {math.degrees(yaw_hold):.1f} deg")
            else:
                HEADING_HOLD_TARGET = None
                print("[ASSIST] Heading hold: OFF")

        # ── Proportional mode toggle (P key) ──────────────────────
        if ord('p') in keys and keys[ord('p')] & p.KEY_WAS_TRIGGERED:
            PROPORTIONAL_MODE = not PROPORTIONAL_MODE
            print(f"[CTRL] Proportional joystick: {'ON' if PROPORTIONAL_MODE else 'OFF (binary)'}")

        # ── Force vector debug visualization toggle (B key) ───────
        if ord('b') in keys and keys[ord('b')] & p.KEY_WAS_TRIGGERED:
            SHOW_FORCE_VECTORS = not SHOW_FORCE_VECTORS
            if not SHOW_FORCE_VECTORS:
                # Remove existing debug lines
                for _fvid in _force_viz_ids.values():
                    try:
                        p.removeUserDebugItem(_fvid)
                    except pybullet.error:
                        pass
                _force_viz_ids.clear()
            print(f"[VIZ] Force vectors: {'ON' if SHOW_FORCE_VECTORS else 'OFF'}")

        # ── Thruster failure toggle (\ key) ───────────────────────
        if ord('\\') in keys and keys[ord('\\')] & p.KEY_WAS_TRIGGERED:
            THRUSTER_FAILURE_ENABLED = not THRUSTER_FAILURE_ENABLED
            if THRUSTER_FAILURE_ENABLED:
                THRUSTER_FAILED = [False] * len(THRUSTERS)
                _thruster_fail_timer = [0.0] * len(THRUSTERS)
                # Fail a random thruster immediately for testing
                import random as _rnd
                fail_idx = _rnd.randint(0, len(THRUSTERS) - 1)
                THRUSTER_FAILED[fail_idx] = True
                print(f"[FAIL] Thruster failure mode: ON — T{fail_idx + 1} FAILED")
            else:
                THRUSTER_FAILED = [False] * len(THRUSTERS)
                print("[FAIL] Thruster failure mode: OFF — all thrusters restored")

        # Emergency surface (key '0') — full heave up, kill horizontals
        if EMERGENCY_SURFACE_KEY in keys and keys[EMERGENCY_SURFACE_KEY] & p.KEY_WAS_TRIGGERED:
            _emergency_surface_active = not _emergency_surface_active
            if _emergency_surface_active:
                print("[⚠️  EMERGENCY SURFACE] Engaging full heave — killing horizontal thrusters")
                for i in range(len(THRUSTERS)):
                    if THRUSTERS[i]["kind"] == "H":
                        thr_cmd[i] = 0.0
                        thr_on[i] = False
                        thr_reverse[i] = False
                    elif THRUSTERS[i]["kind"] == "V":
                        thr_cmd[i] = 1.0
                        thr_on[i] = True
                        thr_reverse[i] = False
            else:
                print("[✅ EMERGENCY SURFACE] Disengaged — manual control restored")
                for i in range(len(THRUSTERS)):
                    if THRUSTERS[i]["kind"] == "V":
                        thr_cmd[i] = 0.0
                        thr_on[i] = False
                        thr_reverse[i] = False

        # Thrust power level adjustment (+/- keys)
        if ord('=') in keys and keys[ord('=')] & p.KEY_WAS_TRIGGERED:
            THRUST_LEVEL = min(1.0, THRUST_LEVEL + THRUST_LEVEL_STEP)
            print(f"[THR] Power level: {THRUST_LEVEL*100:.0f}%")
        if ord('-') in keys and keys[ord('-')] & p.KEY_WAS_TRIGGERED:
            THRUST_LEVEL = max(0.1, THRUST_LEVEL - THRUST_LEVEL_STEP)
            print(f"[THR] Power level: {THRUST_LEVEL*100:.0f}%")

        # Obstacle manipulation keys
        if obstacles:
            # TAB: cycle selected obstacle
            if 9 in keys and keys[9] & p.KEY_WAS_TRIGGERED:
                obs_idx = (obs_idx + 1) % len(obstacles)
                print(f"[OBS] Selected obstacle {obs_idx+1}/{len(obstacles)}")
            obs_move_speed = 0.15  # m per step when key held
            sel_oid = obstacles[obs_idx]
            try:
                o_pos, o_quat = p.getBasePositionAndOrientation(sel_oid)
                o_pos = list(o_pos)
                obs_moved = False
                # WASD move in XY plane
                if ord('w') in keys and keys[ord('w')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
                    o_pos[0] += obs_move_speed; obs_moved = True
                if ord('s') in keys and keys[ord('s')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
                    o_pos[0] -= obs_move_speed; obs_moved = True
                if ord('a') in keys and keys[ord('a')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
                    o_pos[1] += obs_move_speed; obs_moved = True
                if ord('d') in keys and keys[ord('d')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
                    o_pos[1] -= obs_move_speed; obs_moved = True
                # Q/E move up/down
                if ord('q') in keys and keys[ord('q')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
                    o_pos[2] += obs_move_speed; obs_moved = True
                if ord('e') in keys and keys[ord('e')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
                    o_pos[2] -= obs_move_speed; obs_moved = True
                if obs_moved:
                    p.resetBasePositionAndOrientation(sel_oid, o_pos, o_quat)
                    p.resetBaseVelocity(sel_oid, [0, 0, 0], [0, 0, 0])
            except pybullet.error:
                pass

        # Camera controls
        cam_changed = False
        _cam_non_zoom_changed = False  # tracks pitch/yaw/pan (exits top-down)
        if ord('j') in keys and keys[ord('j')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            cam_yaw -= CAM_STEP_ANGLE; cam_changed = True; _cam_non_zoom_changed = True
        if ord('l') in keys and keys[ord('l')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            cam_yaw += CAM_STEP_ANGLE; cam_changed = True; _cam_non_zoom_changed = True
        if ord('i') in keys and keys[ord('i')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            cam_pitch += CAM_STEP_ANGLE; cam_changed = True; _cam_non_zoom_changed = True
        if ord('k') in keys and keys[ord('k')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            cam_pitch -= CAM_STEP_ANGLE; cam_changed = True; _cam_non_zoom_changed = True
        if ord('u') in keys and keys[ord('u')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            cam_dist = max(0.5, cam_dist - CAM_STEP_DIST); cam_changed = True
        if ord('o') in keys and keys[ord('o')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            cam_dist = min(6.0, cam_dist + CAM_STEP_DIST); cam_changed = True
        if (not cam_follow) and ord('n') in keys and keys[ord('n')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            cam_target[2] += CAM_STEP_PAN; cam_changed = True
        if (not cam_follow) and ord('m') in keys and keys[ord('m')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            cam_target[2] -= CAM_STEP_PAN; cam_changed = True

        if cam_changed:
            # If the user manually adjusts pitch/yaw/pan, exit top-down mode.
            # Zoom (U/O) is allowed while in top-down — only non-zoom changes exit.
            if _topdown_active and _cam_non_zoom_changed:
                _topdown_active = False
                print("[CAM] Top-down view: OFF (manual camera override)")
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

        # ---- JOYSTICK PANEL INPUT ----
        # When the joystick panel is active, its axes override keyboard thruster
        # commands.  The mixer returns -1, 0, or +1 (thrusters ON/OFF only).
        # A 0.5 s cooldown prevents the thruster from flipping direction faster
        # than real ESC hardware allows.
        # NOTE: We only set thr_cmd here.  The first-order ramp in the thruster
        # force loop (below) handles thr_level smoothly — this models real motor
        # spool-up/down even though the command itself is instantaneous ON/OFF.
        if ENABLE_JOYSTICK_PANEL and HAS_JOYSTICK:
            js = joystick_panel.get_joystick_state()
            if js.get("active", False):
                js_cmds = joystick_panel.mix_joystick_to_thruster_cmds(
                    js, len(THRUSTERS),
                    proportional=PROPORTIONAL_MODE,
                    input_exponent=INPUT_CURVE_EXPONENT,
                    input_deadzone=INPUT_DEADZONE)
                _now = time.monotonic()
                for i in range(len(THRUSTERS)):
                    desired = js_cmds[i]   # -1, 0, or +1
                    # Determine desired sign (-1, 0, +1)
                    if abs(desired) < 0.03:
                        desired_sign = 0
                    elif desired > 0:
                        desired_sign = 1
                    else:
                        desired_sign = -1

                    # Enforce 0.5 s cooldown on sign changes (fwd↔rev)
                    if desired_sign != 0 and _js_last_sign[i] != 0 and desired_sign != _js_last_sign[i]:
                        # Direction reversal requested — check cooldown
                        if (_now - _js_last_switch_t[i]) < JOYSTICK_SWITCH_COOLDOWN:
                            # Still in cooldown — command zero (coast) instead
                            desired = 0.0
                            desired_sign = 0
                        else:
                            _js_last_switch_t[i] = _now
                    elif desired_sign != _js_last_sign[i]:
                        _js_last_switch_t[i] = _now

                    _js_last_sign[i] = desired_sign

                    # Set thruster command (ON/OFF only).  thr_level is NOT set
                    # here — the first-order ramp in the force loop handles the
                    # motor spool-up/down realistically.
                    if abs(desired) < 0.03:
                        thr_cmd[i] = 0.0
                        thr_on[i] = False
                        thr_reverse[i] = False
                    else:
                        thr_cmd[i] = desired   # -1 or +1
                        thr_on[i] = True
                        thr_reverse[i] = (desired < 0)

                # ---- HEAVE BUTTONS (▲ UP / ▼ DOWN on panel) ----
                # The heave buttons write +1/-1 to js["heave"], which
                # directly commands T3 (vertical thruster, index 2).
                # This is separate from the mixer — T3 is not in the
                # surge/yaw mix, it's button-controlled only.
                # Skip if emergency surface is active (it controls T3).
                if not _emergency_surface_active:
                    heave_cmd = js.get("heave", 0.0)
                    if len(THRUSTERS) > 2:
                        if abs(heave_cmd) > 0.5:
                            heave_sign = 1.0 if heave_cmd > 0 else -1.0
                            thr_cmd[2] = heave_sign
                            thr_on[2] = True
                            thr_reverse[2] = (heave_sign < 0)
                        else:
                            thr_cmd[2] = 0.0
                            thr_on[2] = False
                            thr_reverse[2] = False

        # Read state (guard against solver loss)
        try:
            base_pos, base_quat = p.getBasePositionAndOrientation(rov)
            lin, ang = p.getBaseVelocity(rov)
        except pybullet.error:
            if not p.isConnected():
                print("\n[ERROR] PyBullet connection lost (possible physics server crash).")
                break
            print("\n[ERROR] Physics solver instability detected (getBaseVelocity failed).")
            print("[HINT] Try: lowering MAX_THRUST_H/MAX_THRUST_V, increasing LIN_DRAG_ANG, or reducing MAX_OMEGA.\n")
            break

        # ---- PANEL-DRIVEN RUNTIME SETTINGS ----
        if ENABLE_JOYSTICK_PANEL and HAS_JOYSTICK and getattr(joystick_panel, "_shared", None) is not None:
            try:
                with joystick_panel._shared.get_lock():
                    _set_thrust = joystick_panel._shared[SET_THRUST_LEVEL]
                    _set_prop = joystick_panel._shared[SET_PROPORTIONAL_MODE] > 0.5
                    _set_depth_hold = joystick_panel._shared[SET_DEPTH_HOLD] > 0.5
                    _set_heading_hold = joystick_panel._shared[SET_HEADING_HOLD] > 0.5
                    _set_cam_follow = joystick_panel._shared[SET_CAM_FOLLOW] > 0.5
                    _set_cam_chase = joystick_panel._shared[SET_CAM_CHASE] > 0.5
                    _set_topdown = joystick_panel._shared[SET_TOPDOWN] > 0.5
                    _set_force_viz = joystick_panel._shared[SET_SHOW_FORCE_VECTORS] > 0.5
                    _set_thr_fail = joystick_panel._shared[SET_THRUSTER_FAILURE] > 0.5
                    _set_emergency = joystick_panel._shared[SET_EMERGENCY_SURFACE] > 0.5
                    _set_reset_cmd = int(round(joystick_panel._shared[CMD_RESET_ROV]))
                    _set_trail = joystick_panel._shared[SET_TRAIL_ENABLED] > 0.5
            except (IndexError, ValueError, OSError):
                _set_thrust = THRUST_LEVEL
                _set_prop = PROPORTIONAL_MODE
                _set_depth_hold = DEPTH_HOLD_ENABLED
                _set_heading_hold = HEADING_HOLD_ENABLED
                _set_cam_follow = cam_follow
                _set_cam_chase = CAM_CHASE_ENABLED
                _set_topdown = _topdown_active
                _set_force_viz = SHOW_FORCE_VECTORS
                _set_thr_fail = THRUSTER_FAILURE_ENABLED
                _set_emergency = _emergency_surface_active
                _set_reset_cmd = _panel_last_reset_cmd
                _set_trail = TRAIL_ENABLED

            THRUST_LEVEL = clamp(float(_set_thrust), 0.1, 1.0)

            if _set_prop != _panel_last_prop_mode:
                PROPORTIONAL_MODE = _set_prop
                _panel_last_prop_mode = _set_prop
                print(f"[CTRL] Proportional joystick: {'ON' if PROPORTIONAL_MODE else 'OFF (binary)'}")

            if _set_depth_hold != _panel_last_depth_hold:
                DEPTH_HOLD_ENABLED = _set_depth_hold
                _panel_last_depth_hold = _set_depth_hold
                if DEPTH_HOLD_ENABLED:
                    current_depth = max(0.0, SURFACE_Z - base_pos[2])
                    DEPTH_HOLD_TARGET = current_depth
                    print(f"[ASSIST] Depth hold: ON — holding {current_depth:.2f}m")
                else:
                    DEPTH_HOLD_TARGET = None
                    print("[ASSIST] Depth hold: OFF")

            if _set_heading_hold != _panel_last_heading_hold:
                HEADING_HOLD_ENABLED = _set_heading_hold
                _panel_last_heading_hold = _set_heading_hold
                if HEADING_HOLD_ENABLED:
                    _, _, yaw_hold = p.getEulerFromQuaternion(base_quat)
                    HEADING_HOLD_TARGET = yaw_hold
                    print(f"[ASSIST] Heading hold: ON — holding {math.degrees(yaw_hold):.1f} deg")
                else:
                    HEADING_HOLD_TARGET = None
                    print("[ASSIST] Heading hold: OFF")

            if _set_cam_follow != _panel_last_cam_follow:
                cam_follow = _set_cam_follow
                _panel_last_cam_follow = _set_cam_follow
                print(f"[CAM] Follow mode: {'ON' if cam_follow else 'OFF'}")

            if _set_cam_chase != _panel_last_cam_chase:
                CAM_CHASE_ENABLED = _set_cam_chase
                _panel_last_cam_chase = _set_cam_chase
                print(f"[CAM] Chase camera: {'ON' if CAM_CHASE_ENABLED else 'OFF'}")

            if _set_topdown != _panel_last_topdown:
                _panel_last_topdown = _set_topdown
                _topdown_active = _set_topdown
                if _topdown_active:
                    _topdown_saved = (cam_dist, cam_pitch, cam_yaw)
                    cam_pitch = -89.9
                    cam_dist = 3.5
                    print("[CAM] Top-down view: ON")
                else:
                    cam_dist, cam_pitch, cam_yaw = _topdown_saved
                    print("[CAM] Top-down view: OFF")
                p.resetDebugVisualizerCamera(cameraDistance=cam_dist, cameraYaw=cam_yaw,
                                             cameraPitch=cam_pitch, cameraTargetPosition=cam_target)

            if _set_force_viz != _panel_last_force_viz:
                SHOW_FORCE_VECTORS = _set_force_viz
                _panel_last_force_viz = _set_force_viz
                if not SHOW_FORCE_VECTORS:
                    for _fvid in _force_viz_ids.values():
                        try:
                            p.removeUserDebugItem(_fvid)
                        except pybullet.error:
                            pass
                    _force_viz_ids.clear()
                print(f"[VIZ] Force vectors: {'ON' if SHOW_FORCE_VECTORS else 'OFF'}")

            if _set_thr_fail != _panel_last_thr_fail:
                THRUSTER_FAILURE_ENABLED = _set_thr_fail
                _panel_last_thr_fail = _set_thr_fail
                if THRUSTER_FAILURE_ENABLED:
                    THRUSTER_FAILED = [False] * len(THRUSTERS)
                    _thruster_fail_timer = [0.0] * len(THRUSTERS)
                    fail_idx = random.randint(0, len(THRUSTERS) - 1)
                    THRUSTER_FAILED[fail_idx] = True
                    print(f"[FAIL] Thruster failure mode: ON — T{fail_idx + 1} FAILED")
                else:
                    THRUSTER_FAILED = [False] * len(THRUSTERS)
                    print("[FAIL] Thruster failure mode: OFF — all thrusters restored")

            if _set_emergency != _panel_last_emergency:
                _panel_last_emergency = _set_emergency
                _emergency_surface_active = _set_emergency
                if _emergency_surface_active:
                    print("[⚠️  EMERGENCY SURFACE] Engaging full heave — killing horizontal thrusters")
                    for i in range(len(THRUSTERS)):
                        if THRUSTERS[i]["kind"] == "H":
                            thr_cmd[i] = 0.0
                            thr_on[i] = False
                            thr_reverse[i] = False
                        elif THRUSTERS[i]["kind"] == "V":
                            thr_cmd[i] = 1.0
                            thr_on[i] = True
                            thr_reverse[i] = False
                else:
                    print("[✅ EMERGENCY SURFACE] Disengaged — manual control restored")
                    for i in range(len(THRUSTERS)):
                        if THRUSTERS[i]["kind"] == "V":
                            thr_cmd[i] = 0.0
                            thr_on[i] = False
                            thr_reverse[i] = False

            if _set_trail != _panel_last_trail:
                TRAIL_ENABLED = _set_trail
                _panel_last_trail = _set_trail
                if not TRAIL_ENABLED:
                    for _tid in _trail_line_ids:
                        try:
                            p.removeUserDebugItem(_tid)
                        except pybullet.error:
                            pass
                    _trail_positions.clear()
                    _trail_line_ids.clear()
                    _trail_last_pos = None
                print(f"[VIZ] Trail: {'ON' if TRAIL_ENABLED else 'OFF'}")

            if _set_reset_cmd > _panel_last_reset_cmd:
                _panel_last_reset_cmd = _set_reset_cmd
                start_pos = [0, 0, 0.60]
                start_rpy = [0, 0, 0]
                p.resetBasePositionAndOrientation(rov, start_pos, p.getQuaternionFromEuler(start_rpy))
                p.resetBaseVelocity(rov, [0, 0, 0], [0, 0, 0])
                LAST_VREL_BODY = None
                LAST_A_BODY = (0.0, 0.0, 0.0)
                LAST_W_BODY = None
                LAST_ALPHA_BODY = (0.0, 0.0, 0.0)
                for _tid in _trail_line_ids:
                    try:
                        p.removeUserDebugItem(_tid)
                    except pybullet.error:
                        pass
                _trail_positions.clear()
                _trail_line_ids.clear()
                _trail_last_pos = None
                print(f"[SIM] Reset ROV to {start_pos} facing forward (yaw={start_rpy[2]:.0f}°)")

        # Auto-follow: keep camera tracking the ROV (every frame for smooth visuals)
        if cam_follow:
            follow_lead = 0.3
            cam_target[0] = base_pos[0] + follow_lead
            cam_target[1] = base_pos[1]
            cam_target[2] = base_pos[2] + CAM_FOLLOW_Z_OFFSET

            # ── Chase camera: auto-orbit behind ROV heading ──
            if CAM_CHASE_ENABLED:
                _, _, yaw_chase = p.getEulerFromQuaternion(base_quat)
                # Camera yaw: PyBullet camera yaw 0° = looking along +X.
                # ROV yaw 0° = facing +X.  Camera behind ROV = yaw + 180°.
                desired_cam_yaw = math.degrees(yaw_chase) + 180.0
                # Smooth interpolation (handle wrapping at ±180°)
                diff = desired_cam_yaw - cam_yaw
                while diff > 180.0:
                    diff -= 360.0
                while diff < -180.0:
                    diff += 360.0
                cam_yaw += diff * CAM_CHASE_SMOOTH

            p.resetDebugVisualizerCamera(cameraDistance=cam_dist, cameraYaw=cam_yaw, cameraPitch=cam_pitch, cameraTargetPosition=cam_target)

        # ── Proximity warning ────────────────────────────────────
        if PROXIMITY_WARN_ENABLED:
            dist_to_seabed = base_pos[2] - SEABED_Z
            dist_to_surface = SURFACE_Z - base_pos[2]
            _proximity_warn = (dist_to_seabed < PROXIMITY_WARN_DIST or
                               dist_to_surface < PROXIMITY_WARN_DIST)
        else:
            _proximity_warn = False

        # ── Collision detection (throttled to ~2 Hz to avoid IPC overhead) ──
        if COLLISION_FEEDBACK_ENABLED and (sim_step % max(1, int(0.5 / DT))) == 0:
            _contact_pts = p.getContactPoints(bodyA=rov)
            if _contact_pts and (time.monotonic() - _last_collision_log_t) > COLLISION_LOG_COOLDOWN:
                for cp in _contact_pts[:3]:  # log up to 3 contacts
                    _contact_body_b = cp[2]
                    _contact_force = cp[9]
                    if abs(_contact_force) > 0.5:  # ignore trivial contacts
                        print(f"[💥 COLLISION] body={_contact_body_b} force={_contact_force:.1f}N "
                              f"pos=({cp[5][0]:+.2f},{cp[5][1]:+.2f},{cp[5][2]:+.2f})")
                        _last_collision_log_t = time.monotonic()
                        break

        # ── Breadcrumb trail ─────────────────────────────────────
        if TRAIL_ENABLED:
            if _trail_last_pos is None:
                _trail_last_pos = base_pos
            else:
                dx_tr = base_pos[0] - _trail_last_pos[0]
                dy_tr = base_pos[1] - _trail_last_pos[1]
                dz_tr = base_pos[2] - _trail_last_pos[2]
                if (dx_tr*dx_tr + dy_tr*dy_tr + dz_tr*dz_tr) > TRAIL_SPACING * TRAIL_SPACING:
                    # Age-based color: recent=cyan, old=dark
                    n_pts = len(_trail_positions)
                    if n_pts > 0:
                        age_frac = min(1.0, n_pts / TRAIL_MAX_POINTS)
                        color = [
                            TRAIL_COLOR_RECENT[j] * (1 - age_frac) + TRAIL_COLOR_OLD[j] * age_frac
                            for j in range(3)
                        ]
                    else:
                        color = TRAIL_COLOR_RECENT
                    try:
                        lid = p.addUserDebugLine(
                            list(_trail_last_pos), list(base_pos),
                            color, lineWidth=1.5, lifeTime=0)
                        _trail_line_ids.append(lid)
                    except pybullet.error:
                        pass
                    _trail_positions.append(base_pos)
                    _trail_last_pos = base_pos
                    # Ring buffer: remove oldest segments
                    if len(_trail_positions) > TRAIL_MAX_POINTS:
                        try:
                            p.removeUserDebugItem(_trail_line_ids[0])
                        except pybullet.error:
                            pass
                        _trail_positions.pop(0)
                        _trail_line_ids.pop(0)

        # ROV camera preview — rendered from on-board camera with servo tilt,
        # streamed to the Tkinter camera-view window via shared memory.
        if ENABLE_CAMERA_PREVIEW and (sim_step % prev_every) == 0:
            # --- Servo tilt: rotate fwd/up around camera's lateral (right) axis ---
            # cam_tilt is -1..+1 from the joystick panel slider
            _cam_tilt_norm = 0.0
            if ENABLE_JOYSTICK_PANEL and HAS_JOYSTICK:
                try:
                    _cam_tilt_norm = joystick_panel.get_joystick_state().get("cam_tilt", 0.0)
                except (IndexError, ValueError, OSError):
                    pass
            tilt_rad = _cam_tilt_norm * deg2rad(CAMERA_SERVO_MAX_DEG)

            # Servo tilt axis = lateral axis perpendicular to fwd AND body-Z.
            # Body-Z (0,0,1) maps to world-Z through body rotation, so rotating
            # around fwd×(0,0,1) tilts the camera vertically (up/down) in world.
            _body_up = (0.0, 0.0, 1.0)
            _cr = vcross(cam_fwd_body, _body_up)
            _cr = vnorm(_cr)

            # Rodrigues rotation of fwd and up around _cr by tilt_rad
            _ct = math.cos(tilt_rad)
            _st = math.sin(tilt_rad)

            def _rot_around(v, axis, ct, st):
                # v*cos + (axis×v)*sin + axis*(axis·v)*(1-cos)
                dot = axis[0]*v[0] + axis[1]*v[1] + axis[2]*v[2]
                cx = axis[1]*v[2] - axis[2]*v[1]
                cy = axis[2]*v[0] - axis[0]*v[2]
                cz = axis[0]*v[1] - axis[1]*v[0]
                return (v[0]*ct + cx*st + axis[0]*dot*(1-ct),
                        v[1]*ct + cy*st + axis[1]*dot*(1-ct),
                        v[2]*ct + cz*st + axis[2]*dot*(1-ct))

            tilted_fwd = _rot_around(cam_fwd_body, _cr, _ct, _st)
            tilted_up  = _rot_around(_body_up,     _cr, _ct, _st)

            # Camera pose in WORLD
            cam_pos_world = p.rotateVector(base_quat, cam_pos_body)
            cam_pos_world = (base_pos[0] + cam_pos_world[0], base_pos[1] + cam_pos_world[1], base_pos[2] + cam_pos_world[2])

            # ── Camera shake (speed-dependent vibration) ─────────
            if CAMERA_SHAKE_ENABLED:
                spd_shake = vmag(lin)
                shake_amp = CAMERA_SHAKE_INTENSITY * spd_shake
                if shake_amp > 1e-5:
                    cam_pos_world = (
                        cam_pos_world[0] + random.gauss(0, shake_amp),
                        cam_pos_world[1] + random.gauss(0, shake_amp),
                        cam_pos_world[2] + random.gauss(0, shake_amp * 0.5),
                    )

            cam_fwd_world = p.rotateVector(base_quat, tilted_fwd)
            cam_up_world  = p.rotateVector(base_quat, tilted_up)

            cam_target_pt = (cam_pos_world[0] + cam_fwd_world[0],
                             cam_pos_world[1] + cam_fwd_world[1],
                             cam_pos_world[2] + cam_fwd_world[2])

            view = p.computeViewMatrix(cam_pos_world, cam_target_pt, cam_up_world)
            # Use a slightly larger near plane for the onboard camera to clip
            # any ROV geometry that's right in front of the lens.
            _onboard_proj = p.computeProjectionMatrixFOV(
                fov=CAM_FOV, aspect=float(CAM_PREVIEW_W)/float(CAM_PREVIEW_H),
                nearVal=0.08, farVal=CAM_FAR)
            img = p.getCameraImage(CAM_PREVIEW_W, CAM_PREVIEW_H, viewMatrix=view, projectionMatrix=_onboard_proj, renderer=CAM_RENDERER)
            if img is not None and img[2] is not None:
                rgba = _safe_camera_rgba(img[2], CAM_PREVIEW_W, CAM_PREVIEW_H)

                # ── Depth-dependent camera tint (blue-green at depth) ──
                if rgba is not None and CAMERA_DEPTH_TINT and HAS_NUMPY and depth > 0.5:
                    # Linear ramp: 0% at 0.5m, max at full pool depth
                    _pool_range = max(1.0, SURFACE_Z - SEABED_Z)
                    tint_alpha = min(CAMERA_TINT_MAX_ALPHA, (depth - 0.5) / _pool_range * CAMERA_TINT_MAX_ALPHA)
                    tint = np.array(CAMERA_TINT_COLOR, dtype=np.float32) / 255.0
                    rgb_f = rgba[:, :, :3].astype(np.float32) / 255.0
                    rgb_f = rgb_f * (1.0 - tint_alpha) + tint * tint_alpha
                    rgba_tinted = np.clip(rgb_f * 255.0, 0, 255).astype(np.uint8)
                elif rgba is not None:
                    rgba_tinted = rgba[:, :, :3] if HAS_NUMPY else rgba
                else:
                    rgba_tinted = None

                # ── On-screen display (depth + heading burned into frame) ──
                if rgba_tinted is not None and CAMERA_OSD_ENABLED and HAS_NUMPY:
                    _frame_np = rgba_tinted if HAS_NUMPY else None
                    if _frame_np is not None:
                        _, _, yaw_osd = p.getEulerFromQuaternion(base_quat)
                        heading_deg = math.degrees(yaw_osd) % 360.0
                        # Simple text: burn a dark rectangle + bright pixels as a
                        # minimal OSD.  We draw a thin bar at the top.
                        # Bar background (dark)
                        _frame_np[:14, :, :] = (_frame_np[:14, :, :].astype(np.float32) * 0.3).astype(np.uint8)
                        # We can't easily draw text without PIL/cv2, so we encode
                        # depth and heading as a coloured bar:
                        # - Green bar width = depth fraction of full frame width
                        # - Heading: small orange marker at heading/360 × width
                        depth_frac = min(1.0, depth / max(0.1, SURFACE_Z - SEABED_Z))
                        bar_w = int(depth_frac * CAM_PREVIEW_W)
                        _frame_np[2:6, 2:max(3, bar_w), 1] = 200  # green depth bar
                        heading_px = int((heading_deg / 360.0) * CAM_PREVIEW_W)
                        heading_px = clamp(heading_px, 2, CAM_PREVIEW_W - 3)
                        _frame_np[8:12, heading_px-2:heading_px+2, 0] = 255  # orange heading marker
                        _frame_np[8:12, heading_px-2:heading_px+2, 1] = 140
                        # Proximity warning: red border flash
                        if _proximity_warn:
                            _frame_np[:3, :, 0] = 255   # top red bar
                            _frame_np[:3, :, 1] = 20
                            _frame_np[:3, :, 2] = 20
                            _frame_np[-3:, :, 0] = 255  # bottom red bar
                            _frame_np[-3:, :, 1] = 20
                            _frame_np[-3:, :, 2] = 20
                        rgba_tinted = _frame_np

                # Push RGB frame to joystick panel camera view window
                if rgba_tinted is not None and ENABLE_JOYSTICK_PANEL and HAS_JOYSTICK:
                    try:
                        if HAS_NUMPY:
                            rgb_bytes = rgba_tinted.tobytes()
                        else:
                            # Fallback: strip alpha from flat RGBA bytes
                            flat = bytes(rgba)
                            rgb_bytes = bytes(flat[i] for i in range(len(flat)) if i % 4 != 3)
                        joystick_panel.push_camera_frame(rgb_bytes)
                    except (ValueError, OSError):
                        pass
                # Also show in OpenCV window if available (fallback)
                elif rgba_tinted is not None and cv2 is not None:
                    frame = rgba_tinted[:, :, ::-1]
                    cv2.imshow("ROV Camera", frame)
                    cv2.waitKey(1)
                elif not last_cam_warn:
                    print("[CAM] Camera preview enabled but no display method available.")
                    print("      Start joystick panel or install opencv-python.")
                    last_cam_warn = True

        # ── Screen recording ──────────────────────────────────────────
        # Check if the panel's REC button was toggled.
        _rec_want = False
        if HAS_JOYSTICK and ENABLE_JOYSTICK_PANEL:
            try:
                _rec_want = joystick_panel.is_recording()
            except (IndexError, ValueError, OSError):
                pass

        # Panel dimensions for recording (shared with joystick_panel)
        _PANEL_W = CTRL_W
        _PANEL_H = CTRL_H

        if _rec_want and not _rec_active:
            # --- START recording ---
            if cv2 is None or not HAS_NUMPY:
                print("[REC] ⚠️  Recording requires opencv-python and numpy. Install with:")
                print("      pip install opencv-python numpy")
                try:
                    with joystick_panel._shared.get_lock():
                        joystick_panel._shared[REC_FLAG] = 0.0
                        joystick_panel._shared[REC_STATUS] = REC_STATUS_MISSING_DEPS
                except (IndexError, ValueError, OSError):
                    pass
            else:
                try:
                    with joystick_panel._shared.get_lock():
                        joystick_panel._shared[REC_STATUS] = REC_STATUS_OK
                except (IndexError, ValueError, OSError):
                    pass
                _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                _rec_path = os.path.join(REC_SAVE_DIR, f"rov_recording_{_ts}.mp4")
                # Output: 3D view (left) + full controller panel (right)
                _rec_out_w = REC_3D_W + _PANEL_W
                _rec_out_h = max(REC_3D_H, _PANEL_H)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                _rec_writer = cv2.VideoWriter(_rec_path, fourcc, REC_FPS,
                                              (_rec_out_w, _rec_out_h))
                if _rec_writer.isOpened():
                    _rec_active = True
                    _rec_frame_count = 0
                    _rec_panel_seq = 0
                    print(f"[REC] ⏺  Recording started → {_rec_path}")
                    print(f"[REC]    Output: {_rec_out_w}×{_rec_out_h} @ {REC_FPS}fps")
                    print(f"[REC]    Layout: [3D view {REC_3D_W}×{REC_3D_H} | Controller {_PANEL_W}×{_PANEL_H}]")
                else:
                    print(f"[REC] ⚠️  Failed to open video writer for {_rec_path}")
                    _rec_writer = None
                    try:
                        with joystick_panel._shared.get_lock():
                            joystick_panel._shared[REC_FLAG] = 0.0
                            joystick_panel._shared[REC_STATUS] = REC_STATUS_WRITER_OPEN_FAILED
                    except (IndexError, ValueError, OSError):
                        pass

        elif _rec_active and not _rec_want:
            # --- STOP recording ---
            if _rec_writer is not None:
                _rec_writer.release()
                _dur = _rec_frame_count / max(1, REC_FPS)
                print(f"[REC] ⏹  Recording saved: {_rec_path}")
                print(f"[REC]    {_rec_frame_count} frames, ~{_dur:.1f}s duration")
                _rec_writer = None
            try:
                with joystick_panel._shared.get_lock():
                    joystick_panel._shared[REC_STATUS] = REC_STATUS_OK
            except (AttributeError, IndexError, ValueError, OSError):
                pass
            _rec_active = False
            _rec_path = None

        # --- Write a frame if recording ---
        if _rec_active and _rec_writer is not None and (sim_step % _rec_every) == 0:
            try:
                # 1) Capture 3D debug view from the current camera angle
                _dbg_view = p.computeViewMatrix(
                    cameraEyePosition=[
                        cam_target[0] + cam_dist * math.cos(math.radians(cam_pitch)) * math.cos(math.radians(cam_yaw)),
                        cam_target[1] + cam_dist * math.cos(math.radians(cam_pitch)) * math.sin(math.radians(cam_yaw)),
                        cam_target[2] + cam_dist * math.sin(math.radians(cam_pitch)),
                    ],
                    cameraTargetPosition=cam_target,
                    cameraUpVector=[0, 0, 1])
                _dbg_proj = p.computeProjectionMatrixFOV(
                    fov=60, aspect=float(REC_3D_W) / float(REC_3D_H),
                    nearVal=0.1, farVal=50.0)
                _dbg_img = p.getCameraImage(
                    REC_3D_W, REC_3D_H,
                    viewMatrix=_dbg_view,
                    projectionMatrix=_dbg_proj,
                    renderer=CAM_RENDERER)
                if _dbg_img is not None and _dbg_img[2] is not None:
                    _view3d_rgba = _safe_camera_rgba(_dbg_img[2], REC_3D_W, REC_3D_H)
                    if _view3d_rgba is not None:
                        _view3d = _view3d_rgba[:, :, :3]  # drop alpha
                    else:
                        _view3d = np.zeros((REC_3D_H, REC_3D_W, 3), dtype=np.uint8)
                else:
                    _view3d = np.zeros((REC_3D_H, REC_3D_W, 3), dtype=np.uint8)

                # 2) Get full controller panel screenshot from shared memory
                _panel_frame = None
                try:
                    _p_seq, _p_raw = joystick_panel.get_panel_frame()
                    if _p_raw is not None and _p_seq > 0:
                        _panel_frame = np.frombuffer(_p_raw, dtype=np.uint8).reshape(
                            _PANEL_H, _PANEL_W, 3)
                        _rec_panel_seq = _p_seq
                except (ValueError, OSError):
                    pass
                if _panel_frame is None:
                    _panel_frame = np.zeros((_PANEL_H, _PANEL_W, 3), dtype=np.uint8)

                # 3) Composite side-by-side: [3D view | controller panel]
                _rec_out_h = max(REC_3D_H, _PANEL_H)
                _rec_out_w = REC_3D_W + _PANEL_W
                _composite = np.zeros((_rec_out_h, _rec_out_w, 3), dtype=np.uint8)
                # Place 3D view on the left (vertically centred)
                _y_off_3d = (_rec_out_h - REC_3D_H) // 2
                _composite[_y_off_3d:_y_off_3d + REC_3D_H, :REC_3D_W] = _view3d
                # Place controller panel on the right (vertically centred)
                _y_off_panel = (_rec_out_h - _PANEL_H) // 2
                _composite[_y_off_panel:_y_off_panel + _PANEL_H, REC_3D_W:] = _panel_frame

                # Draw a small red "⏺ REC" indicator in the top-left corner
                _composite[4:12, 4:14, 0] = 220   # red channel
                _composite[4:12, 4:14, 1] = 30
                _composite[4:12, 4:14, 2] = 30

                # 4) Convert RGB→BGR for cv2 and write
                _bgr = _composite[:, :, ::-1]
                _rec_writer.write(_bgr)
                _rec_frame_count += 1
            except (ValueError, OSError, cv2.error) as _rec_err:
                # Don't crash the sim if recording fails
                if _rec_frame_count == 0:
                    print(f"[REC] ⚠️  Frame capture error: {_rec_err}")
                try:
                    with joystick_panel._shared.get_lock():
                        joystick_panel._shared[REC_STATUS] = REC_STATUS_FRAME_WRITE_FAILED
                except (AttributeError, IndexError, ValueError, OSError):
                    pass

        # If autotest is enabled, perform scheduled simulated actions (no GUI keys needed)
        if AUTOTEST and (sim_step in autotest_schedule):
            act = autotest_schedule[sim_step]
            if act == "t1_on":
                thr_on[0] = True; thr_reverse[0] = False
                thr_cmd[0] = 1.0
                if 0 < len(markers): set_marker(markers[0], True)
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] T1: ON FORWARD")
            elif act == "t1_rev":
                thr_reverse[0] = True
                if thr_on[0]:
                    thr_cmd[0] = -1.0
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] T1: REVERSE")
            elif act == "t1_off":
                thr_on[0] = False; thr_cmd[0] = 0.0; thr_level[0] = 0.0; thr_reverse[0] = False
                if 0 < len(markers): set_marker(markers[0], False)
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] T1: OFF")
            elif act == "t2_on":
                thr_on[1] = True; thr_reverse[1] = False
                thr_cmd[1] = 1.0
                if 1 < len(markers): set_marker(markers[1], True)
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] T2: ON FORWARD")
            elif act == "t2_on_rev":
                thr_on[1] = True; thr_reverse[1] = True
                thr_cmd[1] = -1.0
                if 1 < len(markers): set_marker(markers[1], True)
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] T2: ON REVERSE")
            elif act == "t2_rev":
                thr_reverse[1] = not thr_reverse[1]
                if thr_on[1]:
                    thr_cmd[1] = -1.0 if thr_reverse[1] else 1.0
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] T2: {'REVERSE' if thr_reverse[1] else 'FORWARD'}")
            elif act == "t2_off":
                thr_on[1] = False; thr_cmd[1] = 0.0; thr_level[1] = 0.0; thr_reverse[1] = False
                if 1 < len(markers): set_marker(markers[1], False)
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] T2: OFF")
            elif act == "t3_on":
                thr_on[2] = True; thr_reverse[2] = False
                thr_cmd[2] = 1.0
                if 2 < len(markers): set_marker(markers[2], True)
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] T3 (vertical): ON UP")
            elif act == "t3_rev":
                thr_reverse[2] = True
                if thr_on[2]: thr_cmd[2] = -1.0
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] T3 (vertical): REVERSE DOWN")
            elif act == "t3_off":
                thr_on[2] = False; thr_cmd[2] = 0.0; thr_level[2] = 0.0; thr_reverse[2] = False
                if 2 < len(markers): set_marker(markers[2], False)
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] T3: OFF")
            elif act == "t4_on":
                thr_on[3] = True; thr_reverse[3] = False
                thr_cmd[3] = 1.0
                if 3 < len(markers): set_marker(markers[3], True)
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] T4: ON FORWARD")
            elif act == "t4_rev":
                thr_reverse[3] = True
                if thr_on[3]: thr_cmd[3] = -1.0
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] T4: REVERSE")
            elif act == "t4_off":
                thr_on[3] = False; thr_cmd[3] = 0.0; thr_level[3] = 0.0; thr_reverse[3] = False
                if 3 < len(markers): set_marker(markers[3], False)
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] T4: OFF")
            elif act == "t1t2_on":
                for idx in [0, 1]:
                    thr_on[idx] = True; thr_reverse[idx] = False; thr_cmd[idx] = 1.0
                    if idx < len(markers): set_marker(markers[idx], True)
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] T1+T2: ON FORWARD (differential)")
            elif act == "t1t2_off":
                for idx in [0, 1]:
                    thr_on[idx] = False; thr_cmd[idx] = 0.0; thr_level[idx] = 0.0; thr_reverse[idx] = False
                    if idx < len(markers): set_marker(markers[idx], False)
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] T1+T2: OFF")
            elif act in ("all_off", "all_off2"):
                for idx in range(len(THRUSTERS)):
                    thr_on[idx] = False; thr_cmd[idx] = 0.0; thr_level[idx] = 0.0; thr_reverse[idx] = False
                    if idx < len(markers): set_marker(markers[idx], False)
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] ALL THRUSTERS OFF")
            elif act == "all_on":
                for idx in range(len(THRUSTERS)):
                    thr_on[idx] = True; thr_reverse[idx] = False; thr_cmd[idx] = 1.0
                    if idx < len(markers): set_marker(markers[idx], True)
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] ALL THRUSTERS ON FORWARD")
            elif act == "done":
                print(f"[AUTOTEST t={sim_step*DT:.1f}s] Sequence COMPLETE")
                AUTOTEST = False
                if AUTOTEST_EXIT:
                    print("[AUTOTEST] AUTOTEST_EXIT=1 — exiting simulator.")
                    USER_QUIT = True
                    break

        # Update visuals.
        # Thruster indicators: free-standing bodies repositioned each frame.
        if ENABLE_THRUSTER_ARROWS and thr_indicators:
            update_thruster_indicators(thr_indicators, base_pos, base_quat, thr_level, _proximity_warn)

        # Marker sphere positions (if enabled, updated at vis_every rate)
        _need_vis = (sim_step % vis_every) == 0
        if _need_vis and ENABLE_MARKERS:
            try:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            except pybullet.error:
                pass
            for i, t in enumerate(THRUSTERS):
                if i < len(markers):
                    update_marker_pose(markers[i], base_pos, base_quat, t)
            try:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            except pybullet.error:
                pass

        # ---- UPDATE TIME-VARYING OCEAN CURRENT ----
        sim_t = sim_step * DT
        WATER_CURRENT_WORLD[0] = WATER_CURRENT_BASE[0] + CURRENT_VARIATION_AMP[0] * math.sin(2.0 * math.pi * sim_t / CURRENT_VARIATION_PERIOD[0])
        WATER_CURRENT_WORLD[1] = WATER_CURRENT_BASE[1] + CURRENT_VARIATION_AMP[1] * math.sin(2.0 * math.pi * sim_t / CURRENT_VARIATION_PERIOD[1])
        WATER_CURRENT_WORLD[2] = WATER_CURRENT_BASE[2] + CURRENT_VARIATION_AMP[2] * math.cos(2.0 * math.pi * sim_t / CURRENT_VARIATION_PERIOD[2])

        # ---- DEPTH-DEPENDENT BUOYANCY ----
        depth = max(0.0, SURFACE_Z - base_pos[2])  # depth below surface

        # Submersion factor: smooth transition at surface.
        # ROV collision box half-height ≈ 0.144m. When COM is within one hull
        # half-height of the surface, the ROV is partially submerged.
        _hull_half_z = 0.15  # approximate half-height of ROV hull
        if depth >= _hull_half_z:
            submersion = 1.0  # fully submerged
        elif depth <= 0.0:
            submersion = 0.0  # fully out of water
        else:
            submersion = depth / _hull_half_z  # linear transition

        depth_buoyancy_factor = max(0.5, 1.0 - DEPTH_BUOYANCY_COMPRESSIBILITY * depth)

        # ---- WATER FOG / VISIBILITY ----
        # (Fog effect is minimal in PyBullet — skip the per-frame IPC calls)

        # Forces (hydrostatics + water drag)
        # Apply depth-adjusted buoyancy (only when submerged)
        buoy_force = MASS * GRAVITY * BUOYANCY_SCALE * depth_buoyancy_factor * submersion
        cob_rel_world = p.rotateVector(base_quat, COB_OFFSET_BODY)
        cob_world = (base_pos[0] + cob_rel_world[0],
                     base_pos[1] + cob_rel_world[1],
                     base_pos[2] + cob_rel_world[2])
        p.applyExternalForce(rov, -1, (0.0, 0.0, buoy_force), cob_world, p.WORLD_FRAME)

        apply_ballast(rov, base_pos, base_quat)
        if submersion > 0.01:
            apply_righting_torque(rov, base_quat, ang, submersion)
        F_drag, T_drag, sat_drag = apply_drag(rov, base_pos, base_quat, lin, ang)

        # ── Assist mode forces ──────────────────────────────────
        _depth_hold_force = apply_depth_hold(rov, base_pos, lin)
        _heading_hold_torque = apply_heading_hold(rov, base_quat, ang)

        # Push attitude + telemetry to joystick panel
        if ENABLE_JOYSTICK_PANEL and HAS_JOYSTICK and (sim_step % prev_every) == 0:
            try:
                _att_roll, _att_pitch, _ = p.getEulerFromQuaternion(base_quat)
                _, _, _yaw_telem = p.getEulerFromQuaternion(base_quat)
                _spd_telem = math.sqrt(lin[0]**2 + lin[1]**2 + lin[2]**2)
                with joystick_panel._shared.get_lock():
                    joystick_panel._shared[ROLL_RAD] = _att_roll
                    joystick_panel._shared[PITCH_RAD] = _att_pitch
                    joystick_panel._shared[DEPTH_M] = depth
                    joystick_panel._shared[HEADING_DEG] = math.degrees(_yaw_telem) % 360.0
                    joystick_panel._shared[SPEED_MPS] = _spd_telem
                    joystick_panel._shared[SHM_THRUST_LEVEL] = THRUST_LEVEL
                    joystick_panel._shared[DEPTH_HOLD_ACTIVE] = 1.0 if DEPTH_HOLD_ENABLED else 0.0
                    joystick_panel._shared[HEADING_HOLD_ACTIVE] = 1.0 if HEADING_HOLD_ENABLED else 0.0
                    joystick_panel._shared[CONTROL_MODE] = (
                        CONTROL_MODE_PROPORTIONAL if PROPORTIONAL_MODE else CONTROL_MODE_BINARY
                    )
            except (IndexError, ValueError, OSError):
                pass

        # Compute force/torque budget terms for logging — only when needed
        # (these are used by periodic console log and CSV log)
        _need_budget = (log_every > 0 and (sim_step % log_every) == 0) or (LOG_PHYSICS_DETAILED and log_phys_every > 0 and (sim_step % log_phys_every) == 0)
        if _need_budget:
            W = MASS * GRAVITY
            F_buoy = (0.0, 0.0, buoy_force)  # depth-adjusted buoyancy
            F_ball = (0.0, 0.0, -W * BALLAST_SCALE)
            F_grav = (0.0, 0.0, -W)

            cob_rel_w = p.rotateVector(base_quat, COB_OFFSET_BODY)
            bal_rel_w = p.rotateVector(base_quat, BALLAST_OFFSET_BODY)

            roll, pitch, yaw = p.getEulerFromQuaternion(base_quat)
            inv_q_log = p.invertTransform([0, 0, 0], base_quat)[1]
            w_body_log = p.rotateVector(inv_q_log, ang)
            tx_b = (-RIGHTING_K_RP * roll  - RIGHTING_KD_RP * w_body_log[0]) * submersion
            ty_b = (-RIGHTING_K_RP * pitch - RIGHTING_KD_RP * w_body_log[1]) * submersion
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
            # ── Thruster failure: skip force application if this thruster is failed ──
            if THRUSTER_FAILURE_ENABLED and i < len(THRUSTER_FAILED) and THRUSTER_FAILED[i]:
                thr_level[i] = 0.0
                _thr_efficiency[i] = 0.0
                continue

            # First-order ramp to command with asymmetric time constants
            # Now supports -1 (reverse) to +1 (forward)
            if thr_cmd[i] > thr_level[i]:
                ramp_tau = THRUSTER_TAU_UP
            elif thr_cmd[i] < thr_level[i]:
                ramp_tau = THRUSTER_TAU_DN
            else:
                ramp_tau = THRUSTER_TAU_DN  # default
            thr_level[i] += (DT / max(1e-6, ramp_tau)) * (thr_cmd[i] - thr_level[i])
            thr_level[i] = clamp(thr_level[i], -1.0, 1.0)  # Allow negative (reverse)
            # Flush tiny residual values to exactly 0.0 to avoid stuck negative indicator colors
            if abs(thr_level[i]) < 1e-4:
                thr_level[i] = 0.0
            if abs(thr_level[i]) <= 1e-4:
                continue

            thrust_max = MAX_THRUST_H if t["kind"] == "H" else MAX_THRUST_V
            thrust = thrust_max * thr_level[i] * THRUST_LEVEL  # Apply power level scaling
            # Scale reverse thrust magnitude to be a bit lower than forward
            if thrust < 0.0:
                thrust *= BACKWARDS_THRUST_SCALE

            # Simple empirical thrust loss due to inflow: reduce thrust as the
            # local inflow speed along the propeller axis increases.
            dir_body = t.get("dir", (1.0, 0.0, 0.0))
            speed_along = v_body_for_thr[0]*dir_body[0] + v_body_for_thr[1]*dir_body[1] + v_body_for_thr[2]*dir_body[2]
            loss = THRUSTER_SPEED_LOSS_COEF * abs(speed_along)
            loss = clamp(loss, 0.0, 0.9)
            thrust *= (1.0 - loss)

            _thr_efficiency[i] = abs(thrust) / max(0.01, thrust_max) if abs(thr_level[i]) > 1e-3 else 0.0

            dir_world = p.rotateVector(base_quat, t["dir"])
            force = (dir_world[0] * thrust, dir_world[1] * thrust, dir_world[2] * thrust)

            F_thr_total = (F_thr_total[0] + force[0], F_thr_total[1] + force[1], F_thr_total[2] + force[2])
            rel_pos_world = p.rotateVector(base_quat, t["pos"])
            torque_thr = vcross(rel_pos_world, force)
            T_thr_total = (T_thr_total[0] + torque_thr[0], T_thr_total[1] + torque_thr[1], T_thr_total[2] + torque_thr[2])
            world_pos = (base_pos[0] + rel_pos_world[0],
                         base_pos[1] + rel_pos_world[1],
                         base_pos[2] + rel_pos_world[2])

            p.applyExternalForce(rov, -1, force, world_pos, p.WORLD_FRAME)

        # ── Debug force vector visualization ──────────────────────
        if SHOW_FORCE_VECTORS and (sim_step % vis_every) == 0:
            _fv_scale = FORCE_VECTOR_SCALE
            # Thrust total (orange)
            _fv_tip = (base_pos[0] + F_thr_total[0] * _fv_scale,
                       base_pos[1] + F_thr_total[1] * _fv_scale,
                       base_pos[2] + F_thr_total[2] * _fv_scale)
            _force_viz_ids["thrust"] = p.addUserDebugLine(
                list(base_pos), list(_fv_tip), [1.0, 0.6, 0.1], 3, lifeTime=0,
                replaceItemUniqueId=_force_viz_ids.get("thrust", -1))
            # Drag (blue)
            _fv_tip_d = (base_pos[0] + F_drag[0] * _fv_scale,
                         base_pos[1] + F_drag[1] * _fv_scale,
                         base_pos[2] + F_drag[2] * _fv_scale)
            _force_viz_ids["drag"] = p.addUserDebugLine(
                list(base_pos), list(_fv_tip_d), [0.2, 0.5, 1.0], 2, lifeTime=0,
                replaceItemUniqueId=_force_viz_ids.get("drag", -1))
            # Buoyancy (green, from COB)
            _buoy_tip = (cob_world[0], cob_world[1], cob_world[2] + buoy_force * _fv_scale)
            _force_viz_ids["buoy"] = p.addUserDebugLine(
                list(cob_world), list(_buoy_tip), [0.1, 0.9, 0.3], 2, lifeTime=0,
                replaceItemUniqueId=_force_viz_ids.get("buoy", -1))
            # Current direction (cyan arrow from ROV centre)
            _cur_mag = math.sqrt(WATER_CURRENT_WORLD[0]**2 + WATER_CURRENT_WORLD[1]**2 + WATER_CURRENT_WORLD[2]**2)
            if _cur_mag > 1e-4:
                _cur_sc = 3.0  # scale up current vectors for visibility
                _cur_tip = (base_pos[0] + WATER_CURRENT_WORLD[0] * _cur_sc,
                            base_pos[1] + WATER_CURRENT_WORLD[1] * _cur_sc,
                            base_pos[2] + WATER_CURRENT_WORLD[2] * _cur_sc)
                _force_viz_ids["current"] = p.addUserDebugLine(
                    list(base_pos), list(_cur_tip), [0.0, 0.8, 0.8], 1, lifeTime=0,
                    replaceItemUniqueId=_force_viz_ids.get("current", -1))

        # Structured, per-step physics logging (CSV-like) to help tuning/optimization
        if LOG_PHYSICS_DETAILED and log_phys_every > 0 and (sim_step % log_phys_every) == 0 and _log_file_handle is not None:
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
                    f"{F_buoy[2]:.3f},{F_ball[2]:.3f},"
                    f"{depth:.3f},{depth_buoyancy_factor:.4f},"
                    f"{WATER_CURRENT_WORLD[0]:.4f},{WATER_CURRENT_WORLD[1]:.4f},{WATER_CURRENT_WORLD[2]:.4f},"
                    f"{math.degrees(roll):.2f},{math.degrees(pitch):.2f},{math.degrees(yaw):.2f},"
                    f"\"{thr_levels_str}\""
                )
                # Append obstacle positions + velocities
                obs_csv = ""
                for oid in obstacles:
                    try:
                        opos, _ = p.getBasePositionAndOrientation(oid)
                        ovel, _ = p.getBaseVelocity(oid)
                        obs_csv += f",{opos[0]:.4f},{opos[1]:.4f},{opos[2]:.4f},{ovel[0]:.4f},{ovel[1]:.4f},{ovel[2]:.4f}"
                    except pybullet.error:
                        obs_csv += ",,,,,,"
                line += obs_csv + "\n"
                _log_file_handle.write(line)
            except OSError:
                # Keep logging best-effort; never crash sim for logging errors
                pass

        # Periodic debug logging so we can see what is happening from the terminal output
        if log_every > 0 and (sim_step % log_every) == 0 and _need_budget:
            spd = math.sqrt(lin[0]*lin[0] + lin[1]*lin[1] + lin[2]*lin[2])
            omg = math.sqrt(ang[0]*ang[0] + ang[1]*ang[1] + ang[2]*ang[2])
            
            # Formatted telemetry output
            print(f"\n┌─ t={sim_step*DT:7.2f}s ─────────────────────────────────────────────────┐")
            print(f"│ 📍 Position: ({base_pos[0]:+6.2f}, {base_pos[1]:+6.2f}, {base_pos[2]:+6.2f}) m")
            print(f"│ 🚀 Velocity: ({lin[0]:+6.2f}, {lin[1]:+6.2f}, {lin[2]:+6.2f}) m/s  │v│={spd:5.2f}")
            print(f"│ 🔄 Attitude: RPY=({math.degrees(roll):+7.1f}°, {math.degrees(pitch):+7.1f}°, {math.degrees(yaw):+7.1f}°)")
            print(f"│ ⚙️  Angular: ω=({ang[0]:+6.2f}, {ang[1]:+6.2f}, {ang[2]:+6.2f}) rad/s  │ω│={omg:5.2f}")
            
            # Thrusters on one line (with efficiency %)
            thr_status = " ".join([
                f"T{i+1}:{thr_level[i]:+4.2f}({_thr_efficiency[i]*100:.0f}%)" if thr_on[i] else f"T{i+1}:----"
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
            if _proximity_warn:
                warning_flags.append("🔴 PROXIMITY")
            if _emergency_surface_active:
                warning_flags.append("🆘 EMERGENCY SURFACE")
            if DEPTH_HOLD_ENABLED:
                warning_flags.append(f"DH:{DEPTH_HOLD_TARGET:.1f}m f={_depth_hold_force:+.1f}N")
            if HEADING_HOLD_ENABLED:
                warning_flags.append(f"HH:{math.degrees(HEADING_HOLD_TARGET):.0f}° τ={_heading_hold_torque:+.2f}")
            if THRUSTER_FAILURE_ENABLED:
                _fail_str = " ".join([f"T{j+1}:FAIL" for j in range(len(THRUSTER_FAILED)) if THRUSTER_FAILED[j]])
                if _fail_str:
                    warning_flags.append(f"💥 {_fail_str}")
            
            if warning_flags:
                print(f"│ {' | '.join(warning_flags)}")

            if LOG_OBS and obstacles:
                for oi, oid in enumerate(obstacles):
                    try:
                        o_pos, _ = p.getBasePositionAndOrientation(oid)
                        o_lin, _ = p.getBaseVelocity(oid)
                        o_spd = math.sqrt(o_lin[0]**2 + o_lin[1]**2 + o_lin[2]**2)
                        marker = "🎯" if oi == obs_idx else "  "
                        print(f"│ {marker} Obs[{oi+1}/{len(obstacles)}]: pos=({o_pos[0]:+.2f},{o_pos[1]:+.2f},{o_pos[2]:+.2f}) "
                              f"v=({o_lin[0]:+.2f},{o_lin[1]:+.2f},{o_lin[2]:+.2f}) |v|={o_spd:.3f}")
                    except pybullet.error:
                        pass
            
            print(f"└────────────────────────────────────────────────────────────────────┘")

        # Water forces for obstacles (neutral buoyancy + drag) — EVERY step
        # (Must match gravity which PyBullet applies every step; skipping steps
        #  creates a net downward force = weight * (1 - 1/skip_factor))
        apply_obstacle_water_forces(obstacles)

        # ---- HUD UPDATE ----
        if HUD_ENABLED and (sim_step % hud_every) == 0:
            spd_hud = math.sqrt(lin[0]**2 + lin[1]**2 + lin[2]**2)
            roll_h, pitch_h, yaw_h = p.getEulerFromQuaternion(base_quat)
            # Depth below water surface
            pressure_atm = 1.0 + (RHO * GRAVITY * depth) / 101325.0  # atmospheres

            # Build a single combined HUD string (1 IPC call instead of 5)
            thr_str = " ".join([
                f"T{j+1}:{'R' if thr_reverse[j] else 'F'}{abs(thr_level[j]):.0%}" if thr_on[j]
                else f"T{j+1}:OFF" for j in range(len(THRUSTERS))
            ])
            hud_lines = [
                f"Depth: {depth:.1f}m  Pressure: {pressure_atm:.2f} atm",
                f"Speed: {spd_hud:.2f} m/s",
                f"Power: {THRUST_LEVEL*100:.0f}%  {thr_str}",
                f"RPY: {math.degrees(roll_h):+.1f}  {math.degrees(pitch_h):+.1f}  {math.degrees(yaw_h):+.1f}",
            ]
            # Assist mode status
            _assists = []
            if DEPTH_HOLD_ENABLED:
                _assists.append(f"DH:{DEPTH_HOLD_TARGET:.1f}m")
            if HEADING_HOLD_ENABLED:
                _holds_hdg = math.degrees(HEADING_HOLD_TARGET) if HEADING_HOLD_TARGET else 0
                _assists.append(f"HH:{_holds_hdg:.0f}deg")
            if THRUSTER_FAILURE_ENABLED:
                _failed_list = [f"T{j+1}" for j in range(len(THRUSTER_FAILED)) if THRUSTER_FAILED[j]]
                if _failed_list:
                    _assists.append(f"FAIL:{','.join(_failed_list)}")
            if _assists:
                hud_lines.append(" ".join(_assists))
            hud_text = "\n".join(hud_lines)

            # Position near ROV in world space (offset above and to the side)
            hud_pos = [base_pos[0] - 0.5, base_pos[1] + 0.8, base_pos[2] + 0.6]
            # Slight offset for shadow layer (simulates dark outline for readability)
            shadow_pos = [hud_pos[0] + 0.008, hud_pos[1] - 0.008, hud_pos[2] - 0.008]
            try:
                if _hud_shadow_id is not None:
                    _hud_shadow_id = p.addUserDebugText(
                        hud_text, shadow_pos,
                        textColorRGB=[0.0, 0.0, 0.0], textSize=1.15, lifeTime=0,
                        replaceItemUniqueId=_hud_shadow_id)
                hud_items["combined"] = p.addUserDebugText(
                    hud_text, hud_pos,
                    textColorRGB=[1.0, 1.0, 1.0], textSize=1.1, lifeTime=0,
                    replaceItemUniqueId=hud_items["combined"])
            except pybullet.error:
                pass

        sim_step += 1
        _sim_clock += DT
        if not p.isConnected():
            break
        p.stepSimulation()

        # ── Timing metrics ───────────────────────────────────────
        _timing_frame_count += 1
        _timing_now = time.monotonic()
        if SHOW_TIMING_METRICS and (_timing_now - _timing_last_report) >= TIMING_REPORT_INTERVAL:
            _elapsed = _timing_now - _timing_last_report
            _fps = _timing_frame_count / max(0.001, _elapsed)
            _step_ms = (_elapsed / max(1, _timing_frame_count)) * 1000.0
            _realtime_ratio = (_timing_frame_count * DT) / max(0.001, _elapsed)
            print(f"[PERF] {_fps:.1f} FPS | step {_step_ms:.2f}ms | "
                  f"realtime x{_realtime_ratio:.2f} | "
                  f"sim {sim_step * DT:.1f}s")
            _timing_frame_count = 0
            _timing_last_report = _timing_now

        # Real-time pacing: sleep only the remaining time after computation
        if SLEEP_REALTIME:
            _wall_elapsed = time.monotonic() - _wall_t0
            _sleep_needed = _sim_clock - _wall_elapsed
            if _sleep_needed > 0.001:
                time.sleep(_sleep_needed)
        # Flush log file periodically (every ~2 seconds) instead of every write
        _flush_counter += 1
        if _flush_counter >= int(2.0 / DT):
            _flush_counter = 0
            if _log_file_handle is not None:
                try:
                    _log_file_handle.flush()
                except OSError:
                    pass

    teardown_simulation(_rec_active, _rec_writer, _rec_frame_count, _rec_path)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for PyInstaller on Windows
    import traceback as _tb
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SIM] Interrupted by user (Ctrl+C). Exiting.")
    except Exception as _e:
        print(f"[ERROR] Simulator crashed: {_e}")
        _tb.print_exc()
    finally:
        # Ensure PyBullet is disconnected on any exit path
        try:
            p.disconnect()
        except pybullet.error:
            pass
    print("Simulator exited.")

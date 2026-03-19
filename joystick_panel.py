"""
ROV Controller Panel — runs as a separate PROCESS.

On macOS, Tkinter must run on the main thread, and PyBullet's GUI also wants
the main thread.  So this module launches a separate Python process that hosts
the Tkinter window, communicating joystick axes back via shared memory
(multiprocessing.Array).

UI includes camera feed, two draggable joystick knobs, two red
heave buttons (▲ UP / ▼ DOWN), a ⏺ REC / ⏹ STOP button,
and a SETTINGS dialog that controls simulator runtime options
(assist modes, camera behavior, debug toggles, thrust level, reset).

Shared memory layout (32 doubles):
  [0]  surge    (-1..1)  — left stick Y
  [1]  sway     (unused, always 0)
  [2]  heave    (-1/0/+1) — red buttons: ▲ UP = +1, ▼ DN = -1
  [3]  yaw      (-1..1)  — left stick X
  [4]  active   (1.0 = panel open, 0.0 = closed)
  [5]  cam_tilt (-1..1)  — right stick Y
  [6]  roll_rad
  [7]  pitch_rad
  [8]  rec_flag
  [9]  depth_m
  [10] heading_deg
  [11] speed_mps
  [12] thrust_level
  [13] depth_hold_active
  [14] heading_hold_active
  [15] rec_status         — simulator writes recording error code (0 = OK)
  [16] control_mode       — simulator writes 0 = binary, 1 = proportional
    [17] set_thrust_level      — desired global thrust scale (0.1..1.0)
    [18] set_proportional_mode — 0/1
    [19] set_depth_hold        — 0/1
    [20] set_heading_hold      — 0/1
    [21] set_cam_follow        — 0/1
    [22] set_cam_chase         — 0/1
    [23] set_topdown           — 0/1
    [24] set_show_force_viz    — 0/1
    [25] set_thruster_failure  — 0/1
    [26] set_emergency_surface — 0/1
    [27] cmd_reset_rov         — pulse counter
    [28] set_trail_enabled     — 0/1
    [29-31] reserved

Frame buffer (shared between sim and panel process):
  _frame_buf : RawArray of bytes (CAM_W * CAM_H * 3) — RGB pixels
  _frame_seq : Value('i') — frame sequence counter (sim increments, panel reads)
"""

import multiprocessing
import ctypes
import math
import time
import sys
import os

try:
    from debug.runtime_events import RuntimeEventLogger
except ImportError:
    RuntimeEventLogger = None

from sim_shared import (
    ACTIVE,
    CAM_TILT,
    CMD_RESET_ROV,
    CONTROL_MODE,
    CONTROL_MODE_BINARY,
    CTRL_H,
    CTRL_W,
    DEPTH_HOLD_ACTIVE,
    DEPTH_M,
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
    HEADING_DEG,
    HEADING_HOLD_ACTIVE,
    HEAVE,
    PITCH_RAD,
    REC_FLAG,
    REC_STATUS,
    REC_STATUS_OK,
    REC_STATUS_MISSING_DEPS,
    REC_STATUS_WRITER_OPEN_FAILED,
    REC_STATUS_PANEL_CAPTURE_UNAVAILABLE,
    REC_STATUS_FRAME_WRITE_FAILED,
    ROLL_RAD,
    SHARED_SLOT_COUNT,
    SPEED_MPS,
    SURGE,
    SWAY,
    THRUST_LEVEL,
    YAW,
    control_mode_label,
    recording_status_label,
)

# ── Constants ────────────────────────────────────────────────────────
CAM_W = 320
CAM_H = 240
_FRAME_NBYTES = CAM_W * CAM_H * 3   # RGB

# Controller panel dimensions shared with rov_sim recording output.
_PANEL_NBYTES = CTRL_W * CTRL_H * 3  # RGB screenshot of entire controller

# ── Shared memory ────────────────────────────────────────────────────
_shared = None      # multiprocessing.Array('d', SHARED_SLOT_COUNT)
_frame_buf = None   # multiprocessing.RawArray('B', _FRAME_NBYTES)  — onboard camera
_frame_seq = None   # multiprocessing.Value('i')
_panel_buf = None   # multiprocessing.RawArray('B', _PANEL_NBYTES) — full controller screenshot
_panel_seq = None   # multiprocessing.Value('i')
_process = None
_runtime_events = None


def _init_runtime_events():
    global _runtime_events
    if RuntimeEventLogger is None:
        _runtime_events = None
        return
    _runtime_events = RuntimeEventLogger.from_environment("joystick_panel")


def _evt(category, event, **fields):
    if _runtime_events is None:
        return
    try:
        _runtime_events.emit(category, event, **fields)
    except (OSError, ValueError, TypeError):
        pass


def _ensure_shared():
    global _shared, _frame_buf, _frame_seq, _panel_buf, _panel_seq
    if _shared is None:
        _shared = multiprocessing.Array(ctypes.c_double, SHARED_SLOT_COUNT, lock=True)
        for i in range(SHARED_SLOT_COUNT):
            _shared[i] = 0.0
        _shared[REC_STATUS] = REC_STATUS_OK
        _shared[CONTROL_MODE] = CONTROL_MODE_BINARY
        _shared[SET_THRUST_LEVEL] = 1.0
        _shared[SET_CAM_FOLLOW] = 1.0
        _shared[SET_TRAIL_ENABLED] = 0.0
    if _frame_buf is None:
        _frame_buf = multiprocessing.RawArray(ctypes.c_uint8, _FRAME_NBYTES)
    if _frame_seq is None:
        _frame_seq = multiprocessing.Value(ctypes.c_int, 0, lock=True)
    if _panel_buf is None:
        _panel_buf = multiprocessing.RawArray(ctypes.c_uint8, _PANEL_NBYTES)
    if _panel_seq is None:
        _panel_seq = multiprocessing.Value(ctypes.c_int, 0, lock=True)
    _evt("ipc", "shared_ready", slots=SHARED_SLOT_COUNT)


def get_joystick_state():
    """Return a snapshot of the joystick axes (process-safe via shared memory)."""
    if _shared is None:
        return {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0,
                "active": False, "cam_tilt": 0.0}
    try:
        with _shared.get_lock():
            return {
                "surge":    _shared[SURGE],
                "sway":     _shared[SWAY],
                "heave":    _shared[HEAVE],
                "yaw":      _shared[YAW],
                "active":   _shared[ACTIVE] > 0.5,
                "cam_tilt": _shared[CAM_TILT],
            }
    except Exception:
        _evt("ipc", "get_joystick_state_error")
        return {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0,
                "active": False, "cam_tilt": 0.0}


def push_camera_frame(rgb_bytes):
    """Write an RGB frame (bytes, length CAM_W*CAM_H*3) into shared memory."""
    if _frame_buf is None or _frame_seq is None:
        return
    n = min(len(rgb_bytes), _FRAME_NBYTES)
    ctypes.memmove(ctypes.addressof(_frame_buf), rgb_bytes, n)
    with _frame_seq.get_lock():
        _frame_seq.value += 1
        if (_frame_seq.value % 120) == 0:
            _evt("camera", "frame_seq_tick", frame_seq=int(_frame_seq.value))


def is_recording():
    """Return True if the panel's REC button is active."""
    if _shared is None:
        return False
    try:
        with _shared.get_lock():
            return _shared[REC_FLAG] > 0.5
    except Exception:
        return False


def get_panel_frame():
    """Return (seq, rgb_bytes) of the latest controller panel screenshot.

    Returns (0, None) if no frame is available yet.
    rgb_bytes is a bytes object of CTRL_W * CTRL_H * 3 (RGB).
    """
    if _panel_buf is None or _panel_seq is None:
        return 0, None
    try:
        with _panel_seq.get_lock():
            seq = _panel_seq.value
        if seq < 1:
            return 0, None
        raw = bytes(_panel_buf)
        return seq, raw
    except Exception:
        _evt("camera", "get_panel_frame_error")
        return 0, None


def get_recording_status():
    """Return the simulator-published recording status code."""
    if _shared is None:
        return REC_STATUS_OK
    try:
        with _shared.get_lock():
            return _shared[REC_STATUS]
    except Exception:
        _evt("recording", "get_recording_status_error")
        return REC_STATUS_OK


# ── Tkinter GUI (runs inside child process) ──────────────────────────
def _panel_main(shared_arr, frame_buf, frame_seq, panel_buf, panel_seq):
    """Entry point for the child process — minimal controller panel."""
    import tkinter as tk
    from tkinter import ttk

    def _set(idx, val):
        try:
            with shared_arr.get_lock():
                shared_arr[idx] = val
        except Exception:
            pass

    # ── Colours ───────────────────────────────────────────────────
    COL_BG       = "#1a1a1e"
    COL_BODY     = "#2c2d32"
    COL_BODY_HL  = "#3a3b42"
    COL_BEZEL    = "#222228"
    COL_GRIP     = "#252630"
    COL_GRIP_HL  = "#33343c"
    COL_SCREEN   = "#0c2a3a"
    COL_ACCENT   = "#00bbdd"
    COL_ACCENT2  = "#ff6622"
    COL_JOY_BG   = "#101018"
    COL_JOY_RING = "#555560"
    COL_JOY_KNOB = "#3a3b44"
    COL_JOY_DOT  = "#22aacc"
    COL_OK       = "#44cc88"
    COL_WARN     = "#ffcc66"
    COL_BAD      = "#ff8866"

    VID_W, VID_H = CAM_W, CAM_H       # 320×240
    VID_X = (CTRL_W - VID_W) // 2
    VID_Y = 80

    JOY_R = 58
    JOY_L_CX = 90
    JOY_L_CY = VID_Y + VID_H // 2 + 10
    JOY_R_CX = CTRL_W - 90
    JOY_R_CY = VID_Y + VID_H // 2 + 10
    KNOB_R = 18

    # ── Build the window ──────────────────────────────────────────
    root = tk.Tk()
    root.title("ROV Controller")
    root.configure(bg=COL_BG)
    root.resizable(False, False)
    root.geometry(f"{CTRL_W}x{CTRL_H}+30+80")

    canvas = tk.Canvas(root, width=CTRL_W, height=CTRL_H,
                       bg=COL_BG, highlightthickness=0)
    canvas.pack()

    # ── Helper: rounded rectangle ─────────────────────────────────
    def _rrect(c, x1, y1, x2, y2, r, **kw):
        pts = [
            x1 + r, y1,  x2 - r, y1,
            x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2,
            x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r,
            x1, y1 + r, x1, y1,
        ]
        return c.create_polygon(pts, smooth=True, **kw)

    # ── Controller body ───────────────────────────────────────────
    _rrect(canvas, 40, 30, CTRL_W - 40, CTRL_H - 20, 40,
           fill=COL_BODY, outline=COL_BODY_HL, width=2)

    # Grips
    canvas.create_oval(-10, 130, 130, 400,
                       fill=COL_GRIP, outline=COL_GRIP_HL, width=2)
    canvas.create_oval(CTRL_W - 130, 130, CTRL_W + 10, 400,
                       fill=COL_GRIP, outline=COL_GRIP_HL, width=2)

    # Bottom bridge
    canvas.create_arc(130, CTRL_H - 100, CTRL_W - 130, CTRL_H + 50,
                      start=0, extent=180, fill=COL_BODY, outline=COL_BODY_HL,
                      width=2, style="chord")

    # Antennas
    for ax in [CTRL_W // 2 - 40, CTRL_W // 2 + 40]:
        canvas.create_rectangle(ax - 4, 6, ax + 4, 42,
                                fill="#3a4a55", outline="#2a3a44", width=1)
        canvas.create_oval(ax - 7, 0, ax + 7, 14,
                           fill=COL_ACCENT, outline="#008899", width=1)

    # Power LED
    canvas.create_oval(CTRL_W // 2 - 4, 38, CTRL_W // 2 + 4, 46,
                       fill=COL_ACCENT2, outline="#cc4400", width=1)

    # Screen bezel
    _rrect(canvas, VID_X - 18, VID_Y - 12,
           VID_X + VID_W + 18, VID_Y + VID_H + 12, 10,
           fill=COL_BEZEL, outline="#333340", width=2)

    # Video area
    canvas.create_rectangle(VID_X - 2, VID_Y - 2,
                            VID_X + VID_W + 2, VID_Y + VID_H + 2,
                            fill=COL_SCREEN, outline="#1a4455", width=2)

    # ── Joystick pads (static decorations) ────────────────────────
    for jcx, jcy in [(JOY_L_CX, JOY_L_CY), (JOY_R_CX, JOY_R_CY)]:
        hs = JOY_R + 18
        _rrect(canvas, jcx - hs, jcy - hs, jcx + hs, jcy + hs, 10,
               fill=COL_JOY_BG, outline="#222230", width=2)
        canvas.create_oval(jcx - JOY_R - 6, jcy - JOY_R - 6,
                           jcx + JOY_R + 6, jcy + JOY_R + 6,
                           fill=COL_JOY_RING, outline="#444450", width=1)
        canvas.create_oval(jcx - JOY_R, jcy - JOY_R,
                           jcx + JOY_R, jcy + JOY_R,
                           fill=COL_JOY_BG, outline="#282830", width=1)
        canvas.create_line(jcx - JOY_R + 8, jcy, jcx + JOY_R - 8, jcy,
                           fill="#222230", width=1, dash=(2, 4))
        canvas.create_line(jcx, jcy - JOY_R + 8, jcx, jcy + JOY_R - 8,
                           fill="#222230", width=1, dash=(2, 4))

    # ── Angular zone boundary notches ───────────────────────────────
    # Small tick marks at the 8 zone boundaries (every 45° offset by 22.5°)
    # to show where the stick switches between thruster combinations.
    # The dead-zone circle (15% radius) is also drawn.
    DEAD_FRAC  = 0.15           # matches mixer dead zone
    NOTCH_COL  = "#444450"      # subtle grey
    ZONE_COL   = "#333340"      # even subtler for zone lines
    NOTCH_INNER = 0.18          # zone lines start just outside dead zone
    NOTCH_OUTER = 0.92          # zone lines end near rim

    for jcx, jcy in [(JOY_L_CX, JOY_L_CY), (JOY_R_CX, JOY_R_CY)]:
        # Dead-zone circle
        dr = int(JOY_R * DEAD_FRAC)
        canvas.create_oval(jcx - dr, jcy - dr, jcx + dr, jcy + dr,
                           outline=NOTCH_COL, width=1, dash=(2, 3))
        # Zone boundary lines at 22.5°, 67.5°, 112.5°, ... from up (clockwise)
        # In canvas coords: up = -Y, right = +X
        for zone_deg in [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]:
            rad = math.radians(zone_deg)
            # angle measured clockwise from up: dx=sin(a), dy=-cos(a)
            dx = math.sin(rad)
            dy = -math.cos(rad)
            r_in  = JOY_R * NOTCH_INNER
            r_out = JOY_R * NOTCH_OUTER
            canvas.create_line(
                jcx + dx * r_in,  jcy + dy * r_in,
                jcx + dx * r_out, jcy + dy * r_out,
                fill=ZONE_COL, width=1)

    # Decorative buttons (left side only — right side has heave buttons)
    for bx, by in [(170, VID_Y + VID_H + 10), (190, VID_Y + VID_H + 28)]:
        canvas.create_oval(bx - 6, by - 6, bx + 6, by + 6,
                           fill="#333340", outline="#444450", width=1)

    # ── Operator legend ─────────────────────────────────────────────
    canvas.create_text(56, 56,
                       text="L: SURGE/YAW", fill="#90c8ff",
                       font=("Courier", 8, "bold"), anchor="w")
    canvas.create_text(56, 72,
                       text="R: CAM/YAW", fill="#90c8ff",
                       font=("Courier", 8, "bold"), anchor="w")
    canvas.create_text(56, 88,
                       text="W/S or UP/DN: HEAVE", fill="#ffb3b3",
                       font=("Courier", 8, "bold"), anchor="w")
    canvas.create_text(56, 104,
                       text="SPACE: REC   ESC: CLOSE", fill="#ffe6a3",
                       font=("Courier", 8, "bold"), anchor="w")

    _left_axes_txt = canvas.create_text(JOY_L_CX, JOY_L_CY + JOY_R + 18,
                                        text="S:+0.00 Y:+0.00", fill="#88aacc",
                                        font=("Courier", 8, "bold"), anchor="center")
    _right_axes_txt = canvas.create_text(JOY_R_CX, JOY_R_CY + JOY_R + 18,
                                         text="YT:+0.00", fill="#88aacc",
                                         font=("Courier", 8, "bold"), anchor="center")

    # ── ATTITUDE INDICATOR (artificial horizon) ───────────────────
    # Small circular widget between screen and left stick area.
    # Reads roll/pitch from shared memory [6],[7] (written by sim).
    _ATT_CX = 170      # center X
    _ATT_CY = VID_Y + VID_H + 52  # center Y (below decorative buttons)
    _ATT_R  = 20       # radius

    # Background ring
    canvas.create_oval(_ATT_CX - _ATT_R - 2, _ATT_CY - _ATT_R - 2,
                       _ATT_CX + _ATT_R + 2, _ATT_CY + _ATT_R + 2,
                       fill="#101018", outline="#444450", width=1)
    # Sky half (blue)
    _att_sky = canvas.create_arc(
        _ATT_CX - _ATT_R, _ATT_CY - _ATT_R,
        _ATT_CX + _ATT_R, _ATT_CY + _ATT_R,
        start=0, extent=180, fill="#1a3a55", outline="")
    # Ground half (brown)
    _att_gnd = canvas.create_arc(
        _ATT_CX - _ATT_R, _ATT_CY - _ATT_R,
        _ATT_CX + _ATT_R, _ATT_CY + _ATT_R,
        start=180, extent=180, fill="#3a2a1a", outline="")
    # Horizon line (rotated by roll, offset by pitch)
    _att_horizon = canvas.create_line(
        _ATT_CX - _ATT_R + 4, _ATT_CY,
        _ATT_CX + _ATT_R - 4, _ATT_CY,
        fill="#bbddaa", width=2)
    # Center dot (aircraft wings reference)
    canvas.create_oval(_ATT_CX - 2, _ATT_CY - 2,
                       _ATT_CX + 2, _ATT_CY + 2,
                       fill=COL_ACCENT, outline="")
    # Wing lines
    canvas.create_line(_ATT_CX - 10, _ATT_CY, _ATT_CX - 4, _ATT_CY,
                       fill=COL_ACCENT, width=1)
    canvas.create_line(_ATT_CX + 4, _ATT_CY, _ATT_CX + 10, _ATT_CY,
                       fill=COL_ACCENT, width=1)

    def _update_attitude():
        """Update attitude indicator from shared memory roll/pitch."""
        try:
            with shared_arr.get_lock():
                roll_r  = shared_arr[ROLL_RAD]
                pitch_r = shared_arr[PITCH_RAD]
        except Exception:
            return
        # Roll rotates the horizon line; pitch shifts it vertically
        # Clamp display to ±30° for readability
        roll_deg  = max(-30, min(30, math.degrees(roll_r)))
        pitch_deg = max(-30, min(30, math.degrees(pitch_r)))
        r_rad = math.radians(roll_deg)
        # Horizon line endpoints, rotated by roll, offset by pitch
        pitch_px = pitch_deg / 30.0 * _ATT_R * 0.6  # pixels of shift
        hl = _ATT_R - 4
        x1 = _ATT_CX - hl * math.cos(r_rad)
        y1 = _ATT_CY + hl * math.sin(r_rad) + pitch_px
        x2 = _ATT_CX + hl * math.cos(r_rad)
        y2 = _ATT_CY - hl * math.sin(r_rad) + pitch_px
        canvas.coords(_att_horizon, x1, y1, x2, y2)

    # ── TELEMETRY BAR (bottom centre of controller) ─────────────
    # Shows depth, heading, speed, power level, and assist mode status.
    # Updated periodically alongside camera/attitude.
    _TELEM_Y = CTRL_H - 48
    _TELEM_CX = CTRL_W // 2

    # Background bar
    _rrect(canvas, 140, _TELEM_Y - 14, CTRL_W - 140, _TELEM_Y + 14, 6,
           fill="#1a1a22", outline="#2a2a35", width=1)

    # Telemetry text items (updated in _update_telemetry)
    _telem_depth = canvas.create_text(_TELEM_CX - 160, _TELEM_Y,
        text="D: --", fill="#44cc88", font=("Courier", 9, "bold"), anchor="w")
    _telem_heading = canvas.create_text(_TELEM_CX - 70, _TELEM_Y,
        text="H: ---°", fill="#44aacc", font=("Courier", 9, "bold"), anchor="w")
    _telem_speed = canvas.create_text(_TELEM_CX + 20, _TELEM_Y,
        text="S: ----", fill="#ccaa44", font=("Courier", 9, "bold"), anchor="w")
    _telem_power = canvas.create_text(_TELEM_CX + 110, _TELEM_Y,
        text="P: ---%", fill="#cc6644", font=("Courier", 9, "bold"), anchor="w")
    # Assist mode indicator (right of power)
    _telem_assist = canvas.create_text(_TELEM_CX + 190, _TELEM_Y,
        text="", fill="#88cc44", font=("Courier", 8, "bold"), anchor="w")
    _telem_mode = canvas.create_text(_TELEM_CX + 245, _TELEM_Y,
        text="BIN", fill="#dddd88", font=("Courier", 8, "bold"), anchor="w")
    _telem_rec = canvas.create_text(_TELEM_CX + 290, _TELEM_Y,
        text="", fill="#ff8866", font=("Courier", 8, "bold"), anchor="w")

    _mode_chip = _rrect(canvas, CTRL_W - 146, 42, CTRL_W - 72, 64, 6,
                        fill="#2d2d35", outline="#555560", width=1)
    _mode_chip_txt = canvas.create_text(CTRL_W - 109, 53,
                                        text="MODE BIN", fill="#dddd88",
                                        font=("Courier", 8, "bold"), anchor="center")
    _assist_chip = _rrect(canvas, CTRL_W - 146, 68, CTRL_W - 72, 90, 6,
                          fill="#2d2d35", outline="#555560", width=1)
    _assist_chip_txt = canvas.create_text(CTRL_W - 109, 79,
                                          text="ASSIST --", fill="#aacc88",
                                          font=("Courier", 8, "bold"), anchor="center")
    _rec_chip = _rrect(canvas, CTRL_W - 146, 94, CTRL_W - 72, 116, 6,
                       fill="#2d2d35", outline="#555560", width=1)
    _rec_chip_txt = canvas.create_text(CTRL_W - 109, 105,
                                       text="REC OK", fill="#99dd99",
                                       font=("Courier", 8, "bold"), anchor="center")

    # ── Recording-error toast overlay ──────────────────────────────
    # Shown for ~2.5 s whenever rec_status first transitions to an error code.
    _toast_bg = _rrect(canvas,
                       VID_X + 16, VID_Y + VID_H // 2 - 22,
                       VID_X + VID_W - 16, VID_Y + VID_H // 2 + 22,
                       8, fill="#3a0e0e", outline="#cc3333", width=2, state="hidden")
    _toast_txt = canvas.create_text(VID_X + VID_W // 2, VID_Y + VID_H // 2,
                                    text="", fill="#ffbbaa",
                                    font=("Helvetica", 9, "bold"), anchor="center",
                                    state="hidden")
    _toast_state = {"prev": REC_STATUS_OK, "cancel_id": None}

    _REC_ERROR_DETAILS = {
        int(REC_STATUS_MISSING_DEPS):              "Recording unavailable\n(cv2 / numpy not installed)",
        int(REC_STATUS_WRITER_OPEN_FAILED):        "Recording failed to open output file",
        int(REC_STATUS_PANEL_CAPTURE_UNAVAILABLE): "Panel capture unavailable\n(Pillow not installed)",
        int(REC_STATUS_FRAME_WRITE_FAILED):        "Recording: frame write error",
    }

    def _show_toast(msg):
        canvas.itemconfig(_toast_txt, text=msg, state="normal")
        canvas.itemconfig(_toast_bg, state="normal")
        canvas.tag_raise(_toast_bg)
        canvas.tag_raise(_toast_txt)
        if _toast_state["cancel_id"] is not None:
            root.after_cancel(_toast_state["cancel_id"])
        _toast_state["cancel_id"] = root.after(2500, _hide_toast)

    def _hide_toast():
        canvas.itemconfig(_toast_bg, state="hidden")
        canvas.itemconfig(_toast_txt, state="hidden")
        _toast_state["cancel_id"] = None

    def _update_telemetry():
        """Read telemetry from shared memory and update display."""
        try:
            with shared_arr.get_lock():
                depth_m = shared_arr[DEPTH_M]
                heading_deg = shared_arr[HEADING_DEG]
                speed_mps = shared_arr[SPEED_MPS]
                thrust_pct = shared_arr[THRUST_LEVEL]
                dh_active = shared_arr[DEPTH_HOLD_ACTIVE] > 0.5
                hh_active = shared_arr[HEADING_HOLD_ACTIVE] > 0.5
                rec_status = shared_arr[REC_STATUS]
                control_mode = shared_arr[CONTROL_MODE]
                surge = shared_arr[SURGE]
                yaw = shared_arr[YAW]
                cam_tilt = shared_arr[CAM_TILT]
        except Exception:
            return
        heading_wrap = heading_deg % 360.0
        canvas.itemconfig(_telem_depth, text=f"D:{depth_m:4.1f}m")
        canvas.itemconfig(_telem_heading, text=f"H:{heading_wrap:5.1f}\u00b0")
        canvas.itemconfig(_telem_speed, text=f"S:{speed_mps:4.2f}")
        canvas.itemconfig(_telem_power, text=f"P:{thrust_pct * 100:3.0f}%")
        assists = []
        if dh_active:
            assists.append("DH")
        if hh_active:
            assists.append("HH")
        assist_text = " ".join(assists) if assists else "--"
        mode_text = control_mode_label(control_mode)
        rec_text = recording_status_label(rec_status)

        canvas.itemconfig(_telem_assist, text=assist_text)
        canvas.itemconfig(_telem_mode, text=mode_text)
        canvas.itemconfig(_telem_rec, text=rec_text)

        canvas.itemconfig(_left_axes_txt, text=f"S:{surge:+.2f} Y:{yaw:+.2f}")
        canvas.itemconfig(_right_axes_txt, text=f"YT:{cam_tilt:+.2f}")

        if mode_text == "PROP":
            canvas.itemconfig(_mode_chip, fill="#304850", outline="#44aacc")
            canvas.itemconfig(_mode_chip_txt, text="MODE PROP", fill="#88ddff")
        else:
            canvas.itemconfig(_mode_chip, fill="#3c3a24", outline="#bbaa55")
            canvas.itemconfig(_mode_chip_txt, text="MODE BIN", fill="#ffdd88")

        if assists:
            canvas.itemconfig(_assist_chip, fill="#2f452f", outline="#66aa66")
            canvas.itemconfig(_assist_chip_txt, text=f"ASSIST {assist_text}", fill="#aaffaa")
        else:
            canvas.itemconfig(_assist_chip, fill="#2d2d35", outline="#555560")
            canvas.itemconfig(_assist_chip_txt, text="ASSIST --", fill="#a0a0aa")

        if rec_status == REC_STATUS_OK:
            canvas.itemconfig(_rec_chip, fill="#2f452f", outline="#66aa66")
            canvas.itemconfig(_rec_chip_txt, text="REC OK", fill="#99dd99")
        else:
            canvas.itemconfig(_rec_chip, fill="#4a2f2f", outline="#cc6666")
            canvas.itemconfig(_rec_chip_txt, text=f"REC {rec_text}", fill="#ffaaaa")

        # Show toast when recording status first transitions to a new error code.
        if rec_status != _toast_state["prev"]:
            _toast_state["prev"] = rec_status
            if rec_status != REC_STATUS_OK:
                msg = _REC_ERROR_DETAILS.get(
                    int(round(rec_status)),
                    f"Recording error ({int(round(rec_status))})")
                _show_toast(msg)

    # ── HEAVE BUTTONS (red, right side) ───────────────────────────
    # Two large red buttons for vertical thruster: ▲ UP and ▼ DOWN.
    # Press-and-hold: writes +1 (up) or -1 (down) to shared[2].
    # Release: writes 0.  Binary ON/OFF only.
    _HEAVE_BTN_W  = 52
    _HEAVE_BTN_H  = 30
    _HEAVE_BTN_R  = 8      # corner radius
    _HEAVE_GAP    = 6       # gap between the two buttons
    _HEAVE_CX     = CTRL_W - 168   # horizontal centre
    _HEAVE_TOP_Y  = VID_Y + VID_H + 4   # top of UP button

    COL_BTN_RED       = "#aa2222"
    COL_BTN_RED_PRESS = "#dd3333"
    COL_BTN_RED_OUT   = "#881818"
    COL_BTN_LABEL     = "#ffcccc"

    # UP button
    _hbtn_up_y1 = _HEAVE_TOP_Y
    _hbtn_up_y2 = _hbtn_up_y1 + _HEAVE_BTN_H
    _hbtn_up = _rrect(canvas,
                       _HEAVE_CX - _HEAVE_BTN_W // 2, _hbtn_up_y1,
                       _HEAVE_CX + _HEAVE_BTN_W // 2, _hbtn_up_y2,
                       _HEAVE_BTN_R,
                       fill=COL_BTN_RED, outline=COL_BTN_RED_OUT, width=2)
    _hbtn_up_txt = canvas.create_text(_HEAVE_CX, (_hbtn_up_y1 + _hbtn_up_y2) // 2,
                                       text="▲ UP", fill=COL_BTN_LABEL,
                                       font=("Helvetica", 10, "bold"))

    # DOWN button
    _hbtn_dn_y1 = _hbtn_up_y2 + _HEAVE_GAP
    _hbtn_dn_y2 = _hbtn_dn_y1 + _HEAVE_BTN_H
    _hbtn_dn = _rrect(canvas,
                       _HEAVE_CX - _HEAVE_BTN_W // 2, _hbtn_dn_y1,
                       _HEAVE_CX + _HEAVE_BTN_W // 2, _hbtn_dn_y2,
                       _HEAVE_BTN_R,
                       fill=COL_BTN_RED, outline=COL_BTN_RED_OUT, width=2)
    _hbtn_dn_txt = canvas.create_text(_HEAVE_CX, (_hbtn_dn_y1 + _hbtn_dn_y2) // 2,
                                       text="▼ DN", fill=COL_BTN_LABEL,
                                       font=("Helvetica", 10, "bold"))

    # Heave button state
    _heave_state = {"up": False, "dn": False}

    def _set_heave_up(active):
        _heave_state["up"] = bool(active)
        canvas.itemconfig(_hbtn_up, fill=COL_BTN_RED_PRESS if active else COL_BTN_RED)
        _heave_update()

    def _set_heave_dn(active):
        _heave_state["dn"] = bool(active)
        canvas.itemconfig(_hbtn_dn, fill=COL_BTN_RED_PRESS if active else COL_BTN_RED)
        _heave_update()

    def _heave_update():
        """Write combined heave to shared memory: +1 up, -1 down, 0 off."""
        if _heave_state["up"] and not _heave_state["dn"]:
            _set(HEAVE, 1.0)
        elif _heave_state["dn"] and not _heave_state["up"]:
            _set(HEAVE, -1.0)
        else:
            _set(HEAVE, 0.0)

    def _in_rect(mx, my, x1, y1, x2, y2):
        return x1 <= mx <= x2 and y1 <= my <= y2

    def _heave_press(event):
        mx, my = event.x, event.y
        ux1 = _HEAVE_CX - _HEAVE_BTN_W // 2
        ux2 = _HEAVE_CX + _HEAVE_BTN_W // 2
        if _in_rect(mx, my, ux1, _hbtn_up_y1, ux2, _hbtn_up_y2):
            _set_heave_up(True)
            return True
        if _in_rect(mx, my, ux1, _hbtn_dn_y1, ux2, _hbtn_dn_y2):
            _set_heave_dn(True)
            return True
        return False

    def _heave_release(event):
        changed = False
        if _heave_state["up"]:
            _set_heave_up(False)
            changed = True
        if _heave_state["dn"]:
            _set_heave_dn(False)
            changed = True
        if changed:
            _heave_update()

    # ── RECORD BUTTON ─────────────────────────────────────────────
    # Toggle button: ⏺ REC (green) / ⏹ STOP (red, pulsing).
    # Writes 1.0 to shared[8] when recording, 0.0 when stopped.
    _REC_BTN_W = 56
    _REC_BTN_H = 22
    _REC_BTN_R = 6
    _REC_CX = CTRL_W // 2
    _REC_CY = VID_Y + VID_H + 16  # just below screen bezel

    COL_REC_IDLE     = "#226633"
    COL_REC_IDLE_OUT = "#184422"
    COL_REC_ACTIVE   = "#cc2222"
    COL_REC_ACT_OUT  = "#881818"
    COL_REC_LABEL    = "#ccffcc"
    COL_REC_ACT_LBL  = "#ffcccc"

    _rec_btn = _rrect(canvas,
                      _REC_CX - _REC_BTN_W // 2, _REC_CY - _REC_BTN_H // 2,
                      _REC_CX + _REC_BTN_W // 2, _REC_CY + _REC_BTN_H // 2,
                      _REC_BTN_R,
                      fill=COL_REC_IDLE, outline=COL_REC_IDLE_OUT, width=2)
    _rec_btn_txt = canvas.create_text(_REC_CX, _REC_CY,
                                      text="⏺ REC", fill=COL_REC_LABEL,
                                      font=("Helvetica", 9, "bold"))

    _rec_state = {"on": False, "flash": False}

    def _rec_toggle():
        _rec_state["on"] = not _rec_state["on"]
        if _rec_state["on"]:
            _set(REC_FLAG, 1.0)
            canvas.itemconfig(_rec_btn, fill=COL_REC_ACTIVE, outline=COL_REC_ACT_OUT)
            canvas.itemconfig(_rec_btn_txt, text="⏹ STOP", fill=COL_REC_ACT_LBL)
            _rec_flash()
        else:
            _set(REC_FLAG, 0.0)
            canvas.itemconfig(_rec_btn, fill=COL_REC_IDLE, outline=COL_REC_IDLE_OUT)
            canvas.itemconfig(_rec_btn_txt, text="⏺ REC", fill=COL_REC_LABEL)

    def _rec_flash():
        """Pulse the record button red/dark while recording."""
        if not _rec_state["on"]:
            return
        _rec_state["flash"] = not _rec_state["flash"]
        col = COL_REC_ACTIVE if _rec_state["flash"] else "#881818"
        canvas.itemconfig(_rec_btn, fill=col)
        root.after(500, _rec_flash)

    def _rec_press(event):
        """Check if click is on the record button."""
        mx, my = event.x, event.y
        rx1 = _REC_CX - _REC_BTN_W // 2
        rx2 = _REC_CX + _REC_BTN_W // 2
        ry1 = _REC_CY - _REC_BTN_H // 2
        ry2 = _REC_CY + _REC_BTN_H // 2
        if _in_rect(mx, my, rx1, ry1, rx2, ry2):
            _rec_toggle()
            return True
        return False

    # ── Settings dialog (single source of runtime settings) ───────
    _settings_state = {
        "win": None,
        "reset_seq": 0,
    }

    _set_thrust_var = tk.DoubleVar(value=1.0)
    _set_prop_var = tk.BooleanVar(value=False)
    _set_depth_hold_var = tk.BooleanVar(value=False)
    _set_heading_hold_var = tk.BooleanVar(value=False)
    _set_cam_follow_var = tk.BooleanVar(value=True)
    _set_cam_chase_var = tk.BooleanVar(value=False)
    _set_topdown_var = tk.BooleanVar(value=False)
    _set_force_viz_var = tk.BooleanVar(value=False)
    _set_thr_fail_var = tk.BooleanVar(value=False)
    _set_emergency_var = tk.BooleanVar(value=False)
    _set_trail_var = tk.BooleanVar(value=False)

    def _publish_settings_to_shared():
        try:
            with shared_arr.get_lock():
                shared_arr[SET_THRUST_LEVEL] = float(_set_thrust_var.get())
                shared_arr[SET_PROPORTIONAL_MODE] = 1.0 if _set_prop_var.get() else 0.0
                shared_arr[SET_DEPTH_HOLD] = 1.0 if _set_depth_hold_var.get() else 0.0
                shared_arr[SET_HEADING_HOLD] = 1.0 if _set_heading_hold_var.get() else 0.0
                shared_arr[SET_CAM_FOLLOW] = 1.0 if _set_cam_follow_var.get() else 0.0
                shared_arr[SET_CAM_CHASE] = 1.0 if _set_cam_chase_var.get() else 0.0
                shared_arr[SET_TOPDOWN] = 1.0 if _set_topdown_var.get() else 0.0
                shared_arr[SET_SHOW_FORCE_VECTORS] = 1.0 if _set_force_viz_var.get() else 0.0
                shared_arr[SET_THRUSTER_FAILURE] = 1.0 if _set_thr_fail_var.get() else 0.0
                shared_arr[SET_EMERGENCY_SURFACE] = 1.0 if _set_emergency_var.get() else 0.0
                shared_arr[SET_TRAIL_ENABLED] = 1.0 if _set_trail_var.get() else 0.0
        except Exception:
            pass

    def _send_reset_command():
        _settings_state["reset_seq"] += 1
        try:
            with shared_arr.get_lock():
                shared_arr[CMD_RESET_ROV] = float(_settings_state["reset_seq"])
        except Exception:
            pass

    def _new_check(parent, text, var):
        return tk.Checkbutton(
            parent,
            text=text,
            variable=var,
            command=_publish_settings_to_shared,
            bg="#1c2130",
            fg="#d8e2ff",
            selectcolor="#2b3450",
            activebackground="#252d45",
            activeforeground="#ffffff",
            highlightthickness=0,
            bd=0,
            anchor="w",
            padx=8,
            pady=6,
            font=("Helvetica", 11),
        )

    def _open_settings_window():
        w = _settings_state["win"]
        if w is not None and w.winfo_exists():
            w.lift()
            w.focus_force()
            return

        w = tk.Toplevel(root)
        _settings_state["win"] = w
        w.title("ROV Settings")
        w.geometry("470x560+760+80")
        w.resizable(True, True)
        w.configure(bg="#0f1320")
        style = ttk.Style(w)

        scroll_host = tk.Frame(w, bg="#0f1320")
        scroll_host.pack(fill="both", expand=True)

        scroll_canvas = tk.Canvas(
            scroll_host,
            background="#0f1320",
            highlightthickness=0,
            borderwidth=0,
        )
        scroll_canvas.pack(side="left", fill="both", expand=True)

        # Some Tk builds (notably macOS/Aqua variants) reject certain color options
        # on classic Scrollbar widgets. Keep this path conservative for reliability.
        scroll_bar = ttk.Scrollbar(scroll_host, orient="vertical", command=scroll_canvas.yview)
        scroll_bar.pack(side="right", fill="y")
        scroll_canvas.configure(yscrollcommand=scroll_bar.set)

        outer = tk.Frame(scroll_canvas, bg="#0f1320")
        outer_win = scroll_canvas.create_window((0, 0), window=outer, anchor="nw")

        def _sync_scroll_region(_event=None):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        def _sync_inner_width(event):
            scroll_canvas.itemconfigure(outer_win, width=event.width)

        outer.bind("<Configure>", _sync_scroll_region)
        scroll_canvas.bind("<Configure>", _sync_inner_width)

        def _settings_on_wheel(event):
            if getattr(event, "num", None) == 4:
                scroll_canvas.yview_scroll(-1, "units")
                return "break"
            if getattr(event, "num", None) == 5:
                scroll_canvas.yview_scroll(1, "units")
                return "break"

            delta = getattr(event, "delta", 0)
            if delta:
                units = -1 if delta > 0 else 1
                scroll_canvas.yview_scroll(units, "units")
                return "break"
            return None

        def _bind_settings_wheel(_event=None):
            scroll_canvas.bind_all("<MouseWheel>", _settings_on_wheel)
            scroll_canvas.bind_all("<Button-4>", _settings_on_wheel)
            scroll_canvas.bind_all("<Button-5>", _settings_on_wheel)

        def _unbind_settings_wheel(_event=None):
            scroll_canvas.unbind_all("<MouseWheel>")
            scroll_canvas.unbind_all("<Button-4>")
            scroll_canvas.unbind_all("<Button-5>")

        scroll_canvas.bind("<Enter>", _bind_settings_wheel)
        scroll_canvas.bind("<Leave>", _unbind_settings_wheel)
        w.bind("<Destroy>", _unbind_settings_wheel)

        top_pad = tk.Frame(outer, bg="#0f1320", height=12)
        top_pad.pack(fill="x")

        hero = tk.Frame(outer, bg="#161d31", bd=0, highlightthickness=1, highlightbackground="#2b3756")
        hero.pack(fill="x", padx=14, pady=(0, 12))
        tk.Label(
            hero,
            text="ROV Runtime Console",
            bg="#161d31",
            fg="#e6eeff",
            font=("Helvetica", 16, "bold"),
            anchor="w",
            padx=14,
            pady=12,
        ).pack(fill="x")
        tk.Label(
            hero,
            text="Tune propulsion, assists, camera, and diagnostics live.",
            bg="#161d31",
            fg="#9db0da",
            font=("Helvetica", 10),
            anchor="w",
            padx=14,
            pady=(0, 12),
        ).pack(fill="x")

        def _section(title_text):
            frame = tk.Frame(outer, bg="#1c2130", bd=0, highlightthickness=1, highlightbackground="#2a334b")
            frame.pack(fill="x", padx=14, pady=(0, 10))
            tk.Label(
                frame,
                text=title_text,
                bg="#1c2130",
                fg="#c8d8ff",
                font=("Helvetica", 12, "bold"),
                anchor="w",
                padx=12,
                pady=10,
            ).pack(fill="x")
            body = tk.Frame(frame, bg="#1c2130")
            body.pack(fill="x", padx=10, pady=(0, 10))
            return body

        prop_frame = _section("Propulsion")
        tk.Label(
            prop_frame,
            text="Thrust Level",
            bg="#1c2130",
            fg="#dbe6ff",
            font=("Helvetica", 10, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 4))
        style.configure("Thrust.Horizontal.TScale", troughcolor="#2a334b", background="#1c2130")
        thrust_scale = ttk.Scale(
            prop_frame,
            from_=0.1,
            to=1.0,
            variable=_set_thrust_var,
            command=lambda _v: _publish_settings_to_shared(),
            style="Thrust.Horizontal.TScale",
        )
        thrust_scale.pack(fill="x")
        _thrust_label = tk.Label(
            prop_frame,
            text="100%",
            bg="#1c2130",
            fg="#67d2ff",
            font=("Courier", 11, "bold"),
            anchor="e",
        )
        _thrust_label.pack(fill="x", pady=(2, 6))
        _new_check(prop_frame, "Proportional Joystick Mode", _set_prop_var).pack(fill="x", pady=(2, 0))
        _new_check(prop_frame, "Emergency Surface", _set_emergency_var).pack(fill="x", pady=(2, 0))

        assist_frame = _section("Assist")
        _new_check(assist_frame, "Depth Hold", _set_depth_hold_var).pack(fill="x", pady=(2, 0))
        _new_check(assist_frame, "Heading Hold", _set_heading_hold_var).pack(fill="x", pady=(2, 0))

        cam_frame = _section("Camera")
        _new_check(cam_frame, "Follow ROV", _set_cam_follow_var).pack(fill="x", pady=(2, 0))
        _new_check(cam_frame, "Chase Camera", _set_cam_chase_var).pack(fill="x", pady=(2, 0))
        _new_check(cam_frame, "Top-Down View", _set_topdown_var).pack(fill="x", pady=(2, 0))

        diag_frame = _section("Diagnostics")
        _new_check(diag_frame, "Show Force Vectors", _set_force_viz_var).pack(fill="x", pady=(2, 0))
        _new_check(diag_frame, "Thruster Failure Simulation", _set_thr_fail_var).pack(fill="x", pady=(2, 0))
        _new_check(diag_frame, "Trail Rendering", _set_trail_var).pack(fill="x", pady=(2, 0))

        action_row = tk.Frame(outer, bg="#0f1320")
        action_row.pack(fill="x", padx=14, pady=(0, 14))
        tk.Button(
            action_row,
            text="Reset ROV",
            command=_send_reset_command,
            bg="#2f6aa0",
            fg="#f0f6ff",
            activebackground="#4485c3",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=14,
            pady=8,
            font=("Helvetica", 10, "bold"),
        ).pack(side="left")
        tk.Button(
            action_row,
            text="Close",
            command=w.destroy,
            bg="#30364a",
            fg="#e3e9fb",
            activebackground="#48506a",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=14,
            pady=8,
            font=("Helvetica", 10, "bold"),
        ).pack(side="right")

        def _update_settings_labels():
            _thrust_label.config(text=f"{int(round(_set_thrust_var.get() * 100.0)):d}%")
            _publish_settings_to_shared()
            if w.winfo_exists():
                w.after(120, _update_settings_labels)

        _update_settings_labels()

    _SET_BTN_W = 76
    _SET_BTN_H = 22
    _SET_BTN_R = 6
    _SET_CX = _REC_CX + 82
    _SET_CY = _REC_CY
    _set_btn = _rrect(canvas,
                      _SET_CX - _SET_BTN_W // 2, _SET_CY - _SET_BTN_H // 2,
                      _SET_CX + _SET_BTN_W // 2, _SET_CY + _SET_BTN_H // 2,
                      _SET_BTN_R,
                      fill="#244a66", outline="#2a7399", width=2)
    _set_btn_txt = canvas.create_text(_SET_CX, _SET_CY,
                                      text="SETTINGS", fill="#d3ecff",
                                      font=("Helvetica", 8, "bold"))

    def _settings_press(event):
        mx, my = event.x, event.y
        sx1 = _SET_CX - _SET_BTN_W // 2
        sx2 = _SET_CX + _SET_BTN_W // 2
        sy1 = _SET_CY - _SET_BTN_H // 2
        sy2 = _SET_CY + _SET_BTN_H // 2
        if _in_rect(mx, my, sx1, sy1, sx2, sy2):
            try:
                _open_settings_window()
            except Exception as e:
                print(f"[SETTINGS] Failed to open settings window: {e}")
            return True
        return False

    # ── Joystick knobs (draggable) ────────────────────────────────
    _lknob = canvas.create_oval(JOY_L_CX - KNOB_R, JOY_L_CY - KNOB_R,
                                JOY_L_CX + KNOB_R, JOY_L_CY + KNOB_R,
                                fill=COL_JOY_KNOB, outline="#555560", width=2)
    _lknob_dot = canvas.create_oval(JOY_L_CX - 4, JOY_L_CY - 4,
                                    JOY_L_CX + 4, JOY_L_CY + 4,
                                    fill=COL_JOY_DOT, outline="")
    _rknob = canvas.create_oval(JOY_R_CX - KNOB_R, JOY_R_CY - KNOB_R,
                                JOY_R_CX + KNOB_R, JOY_R_CY + KNOB_R,
                                fill=COL_JOY_KNOB, outline="#555560", width=2)
    _rknob_dot = canvas.create_oval(JOY_R_CX - 4, JOY_R_CY - 4,
                                    JOY_R_CX + 4, JOY_R_CY + 4,
                                    fill=COL_JOY_DOT, outline="")

    # ── Joystick drag logic ───────────────────────────────────────
    _drag = {"active": None}

    def _move_knob(knob_id, dot_id, cx, cy, mx, my, radius):
        dx = mx - cx
        dy = my - cy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > radius:
            dx = dx / dist * radius
            dy = dy / dist * radius
        nx = cx + dx
        ny = cy + dy
        canvas.coords(knob_id,
                       nx - KNOB_R, ny - KNOB_R,
                       nx + KNOB_R, ny + KNOB_R)
        canvas.coords(dot_id,
                       nx - 4, ny - 4, nx + 4, ny + 4)
        ax = dx / radius
        ay = -dy / radius
        if abs(ax) < 0.08:
            ax = 0.0
        if abs(ay) < 0.08:
            ay = 0.0
        return ax, ay

    def _snap_knob(knob_id, dot_id, cx, cy):
        canvas.coords(knob_id,
                       cx - KNOB_R, cy - KNOB_R,
                       cx + KNOB_R, cy + KNOB_R)
        canvas.coords(dot_id,
                       cx - 4, cy - 4, cx + 4, cy + 4)

    # Track left-stick yaw contribution so right stick can add to it
    _shared_yaw_from_left = [0.0]

    def _on_press(event):
        # Check record button first
        if _rec_press(event):
            return
        # Settings button
        if _settings_press(event):
            return
        # Check heave buttons first (they take priority over joystick drag)
        if _heave_press(event):
            _drag["active"] = "heave"
            return
        mx, my = event.x, event.y
        dl = math.sqrt((mx - JOY_L_CX)**2 + (my - JOY_L_CY)**2)
        dr = math.sqrt((mx - JOY_R_CX)**2 + (my - JOY_R_CY)**2)
        if dl < JOY_R + 20:
            _drag["active"] = "left"
            ax, ay = _move_knob(_lknob, _lknob_dot, JOY_L_CX, JOY_L_CY, mx, my, JOY_R)
            _set(SURGE, ay)       # surge = up/down
            _set(YAW, ax)         # yaw   = left/right (positive ax = drag right = yaw right)
        elif dr < JOY_R + 20:
            _drag["active"] = "right"
            ax, ay = _move_knob(_rknob, _rknob_dot, JOY_R_CX, JOY_R_CY, mx, my, JOY_R)
            _set(YAW, _shared_yaw_from_left[0] + ax)  # right stick X adds to yaw
            _set(CAM_TILT, ay)       # cam_tilt = up/down

    def _on_drag(event):
        mx, my = event.x, event.y
        if _drag["active"] == "left":
            ax, ay = _move_knob(_lknob, _lknob_dot, JOY_L_CX, JOY_L_CY, mx, my, JOY_R)
            _set(SURGE, ay)   # surge
            _set(YAW, ax)     # yaw from left stick (positive ax = drag right = yaw right)
            _shared_yaw_from_left[0] = ax
        elif _drag["active"] == "right":
            ax, ay = _move_knob(_rknob, _rknob_dot, JOY_R_CX, JOY_R_CY, mx, my, JOY_R)
            _set(YAW, _shared_yaw_from_left[0] + ax)  # right stick X adds to yaw
            _set(CAM_TILT, ay)       # cam_tilt

    def _on_release(event):
        if _drag["active"] == "heave":
            _heave_release(event)
        elif _drag["active"] == "left":
            _snap_knob(_lknob, _lknob_dot, JOY_L_CX, JOY_L_CY)
            _set(SURGE, 0.0)
            _shared_yaw_from_left[0] = 0.0
            # Preserve right-stick yaw contribution if right stick is also active
            _set(YAW, 0.0)
        elif _drag["active"] == "right":
            _snap_knob(_rknob, _rknob_dot, JOY_R_CX, JOY_R_CY)
            _set(YAW, _shared_yaw_from_left[0])  # Restore to just left-stick yaw
            _set(CAM_TILT, 0.0)
        _drag["active"] = None

    canvas.bind("<ButtonPress-1>", _on_press)
    canvas.bind("<B1-Motion>", _on_drag)
    canvas.bind("<ButtonRelease-1>", _on_release)

    _set(ACTIVE, 1.0)

    # ── Camera frame update (only timer in the panel) ─────────────
    _photo_ref = [None]
    _last_seq = [0]
    _CAM_POLL_MS = 30
    _PANEL_CAP_MS = 50   # panel screenshot interval when recording (~20fps)

    # Try to import Pillow for panel screenshot capture
    try:
        from PIL import ImageGrab
        import numpy as _np
        _has_imagegrab = True
    except ImportError:
        _has_imagegrab = False

    def _capture_panel():
        """Grab a screenshot of this Tkinter window and write into panel_buf."""
        try:
            with shared_arr.get_lock():
                rec_on = shared_arr[REC_FLAG] > 0.5
        except Exception:
            return
        if not rec_on:
            return
        if not _has_imagegrab:
            _set(REC_STATUS, REC_STATUS_PANEL_CAPTURE_UNAVAILABLE)
            return
        try:
            # Check if recording is active
            if not rec_on:
                return
            # Get window position and size on screen
            x = root.winfo_rootx()
            y = root.winfo_rooty()
            w = root.winfo_width()
            h = root.winfo_height()
            if w < 10 or h < 10:
                return
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            # Resize to exactly CTRL_W × CTRL_H if needed
            if img.size != (CTRL_W, CTRL_H):
                img = img.resize((CTRL_W, CTRL_H))
            # Convert RGBA→RGB
            if img.mode == "RGBA":
                img = img.convert("RGB")
            raw = img.tobytes()
            n = min(len(raw), CTRL_W * CTRL_H * 3)
            ctypes.memmove(ctypes.addressof(panel_buf), raw, n)
            with panel_seq.get_lock():
                panel_seq.value += 1
        except Exception:
            _set(REC_STATUS, REC_STATUS_PANEL_CAPTURE_UNAVAILABLE)

    def _update_camera():
        try:
            with frame_seq.get_lock():
                seq = frame_seq.value
        except Exception:
            seq = 0
        if seq > _last_seq[0]:
            _last_seq[0] = seq
            try:
                raw = bytes(frame_buf)
                header = f"P6\n{CAM_W} {CAM_H}\n255\n".encode("ascii")
                ppm_data = header + raw
                photo = tk.PhotoImage(data=ppm_data, format="PPM")
                _photo_ref[0] = photo
                canvas.delete("camframe")
                canvas.create_image(VID_X, VID_Y, anchor="nw", image=photo,
                                    tags="camframe")
            except Exception:
                pass
        # Update attitude indicator alongside camera frame
        _update_attitude()
        # Update telemetry display
        _update_telemetry()
        # Publish panel settings to simulator every UI tick
        _publish_settings_to_shared()
        # Capture panel screenshot for recording (if active)
        _capture_panel()
        root.after(_CAM_POLL_MS, _update_camera)

    _update_camera()

    # ── Close handler ─────────────────────────────────────────────
    def _on_close():
        sw = _settings_state.get("win")
        if sw is not None:
            try:
                sw.destroy()
            except Exception:
                pass
        _set(ACTIVE, 0.0)
        _set(REC_FLAG, 0.0)   # stop recording
        for i in range(6):
            _set(i, 0.0)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()
    _set(ACTIVE, 0.0)
    _set(REC_FLAG, 0.0)   # stop recording
    for i in range(6):
        _set(i, 0.0)


# ── Public API ────────────────────────────────────────────────────────
def start_joystick_panel():
    """Launch the joystick panel in a child process."""
    global _process, _shared
    _ensure_shared()
    _init_runtime_events()
    if _process is not None and _process.is_alive():
        return
    _process = multiprocessing.Process(
        target=_panel_main,
        args=(_shared, _frame_buf, _frame_seq, _panel_buf, _panel_seq),
        daemon=True, name="JoystickPanel")
    _process.start()
    _evt("lifecycle", "panel_process_started", pid=int(_process.pid) if _process.pid else None)
    time.sleep(0.25)


def stop_joystick_panel():
    """Request the panel process to stop (idempotent)."""
    global _process, _shared
    if _shared is not None:
        try:
            with _shared.get_lock():
                _shared[ACTIVE] = 0.0
                _shared[REC_FLAG] = 0.0   # stop recording
                _shared[REC_STATUS] = REC_STATUS_OK
                for i in range(6):
                    _shared[i] = 0.0
        except Exception:
            pass
    if _process is not None:
        try:
            _process.terminate()
            _process.join(timeout=1.0)
            _evt("lifecycle", "panel_process_stopped")
        except Exception:
            pass
        _process = None


# ── Thruster mixer ───────────────────────────────────────────────────
def clamp_val(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _apply_input_curve(val, exponent=1.5, deadzone=0.15):
    """
    Apply deadzone and exponential curve to a joystick axis value.

    Args:
        val: raw axis value (-1..1)
        exponent: curve exponent (>1 = more precision near centre)
        deadzone: fraction of travel treated as zero

    Returns:
        Shaped value (-1..1) with deadzone removed and curve applied.
    """
    if abs(val) < deadzone:
        return 0.0
    # Remove deadzone from the active range
    sign = 1.0 if val > 0 else -1.0
    normalized = (abs(val) - deadzone) / (1.0 - deadzone)
    normalized = min(1.0, max(0.0, normalized))
    # Apply power curve
    shaped = normalized ** exponent
    return sign * shaped


def mix_joystick_to_thruster_cmds(state, n_thrusters, proportional=False,
                                   input_exponent=1.5, input_deadzone=0.15):
    """
    Convert joystick axes to per-thruster commands.

    When proportional=False (default): outputs only -1, 0, or +1 (binary mode).
    When proportional=True: outputs continuous -1..+1 with input curve shaping.

    DDR thruster layout (after GLTF detection + body-frame rotation):
      T1 = rear-right — angled ~40° outboard. cmd=+1 pushes forward+left, yaw-left torque
      T2 = rear-left  — angled ~40° outboard. cmd=+1 pushes forward+right, yaw-right torque
      T3 = vertical (heave) — keyboard-only, not mixed here
      T4 = front centre — nearly pure forward. cmd=+1 pushes forward

    Key insight: ALL three horizontal thrusters have a large forward component
    (T1: 0.77, T2: 0.77, T4: 1.00). Using all three for forward gives 14.1 N
    instead of only 5.6 N from T4 alone — 2.5× more thrust.

    Binary zone mixer:
      surge_cmd  = joystick surge  (-1..+1)
      yaw_cmd    = joystick yaw    (-1..+1, positive = yaw RIGHT)

    The continuous proportional values are computed first, then snapped:
      raw_T1 = surge_cmd - yaw_cmd
      raw_T2 = surge_cmd + yaw_cmd
      raw_T4 = surge_cmd

    Each raw value is then snapped to -1, 0, or +1 (binary mode) or
    clamped to -1..+1 (proportional mode).

    Sign convention (verified from geometry):
      T1 cmd=+1 → yaw torque +1.14 N·m (CCW = yaw LEFT)
      T2 cmd=+1 → yaw torque -1.14 N·m (CW  = yaw RIGHT)
    So to yaw RIGHT (CW, negative τz): decrease T1, increase T2.

    Dead zone at 15% overall stick magnitude.
    T3 (heave) is keyboard-only — always 0 from joystick.
    """
    surge = state.get("surge", 0.0)
    yaw   = state.get("yaw", 0.0)

    DEAD = input_deadzone
    mag = math.sqrt(surge * surge + yaw * yaw)

    cmds = [0.0] * n_thrusters
    if mag < DEAD:
        return cmds

    if proportional:
        # Apply input curves to each axis independently
        surge_shaped = _apply_input_curve(surge, input_exponent, DEAD)
        yaw_shaped = _apply_input_curve(yaw, input_exponent, DEAD)

        # Compute raw proportional values
        raw_t1 = surge_shaped - yaw_shaped
        raw_t2 = surge_shaped + yaw_shaped
        raw_t4 = surge_shaped

        # Clamp to -1..+1
        def _clamp(v):
            return max(-1.0, min(1.0, v))

        if n_thrusters >= 1:
            cmds[0] = _clamp(raw_t1)
        if n_thrusters >= 2:
            cmds[1] = _clamp(raw_t2)
        if n_thrusters >= 4:
            cmds[3] = _clamp(raw_t4)
    else:
        # Binary mode: snap to -1, 0, or +1
        raw_t1 = surge - yaw
        raw_t2 = surge + yaw
        raw_t4 = surge

        SNAP_THRESHOLD = 0.15
        def _snap(val):
            if abs(val) < SNAP_THRESHOLD:
                return 0.0
            return 1.0 if val > 0 else -1.0

        t1 = _snap(raw_t1)
        t2 = _snap(raw_t2)
        t4 = _snap(raw_t4)

        if n_thrusters >= 1:
            cmds[0] = t1
        if n_thrusters >= 2:
            cmds[1] = t2
        if n_thrusters >= 4:
            cmds[3] = t4
    return cmds

"""
ROV Controller Panel — runs as a separate PROCESS.

On macOS, Tkinter must run on the main thread, and PyBullet's GUI also wants
the main thread.  So this module launches a separate Python process that hosts
the Tkinter window, communicating joystick axes back via shared memory
(multiprocessing.Array).

Minimal UI — camera feed, two draggable joystick knobs, two red
heave buttons (▲ UP / ▼ DOWN) for vertical thruster control, and a
⏺ REC / ⏹ STOP button for screen recording.
No text overlays, no status updates, no timers except the camera poll.

Shared memory layout (10 doubles):
  [0] surge    (-1..1)  — left stick Y
  [1] sway     (unused, always 0)
  [2] heave    (-1/0/+1) — red buttons: ▲ UP = +1, ▼ DN = -1
  [3] yaw      (-1..1)  — left stick X  (positive = drag right = yaw right)
  [4] active   (1.0 = panel open, 0.0 = closed)
  [5] cam_tilt (-1..1)  — right stick Y  (camera servo pitch)
  [6] roll_rad           — sim writes ROV roll (radians) for attitude indicator
  [7] pitch_rad          — sim writes ROV pitch (radians) for attitude indicator
  [8] rec_flag           — 1.0 = recording requested, 0.0 = stop/idle
  [9] (reserved)

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

# ── Constants ────────────────────────────────────────────────────────
CAM_W = 320
CAM_H = 240
_FRAME_NBYTES = CAM_W * CAM_H * 3   # RGB

# Controller panel dimensions (must match _panel_main)
CTRL_W = 720
CTRL_H = 480
_PANEL_NBYTES = CTRL_W * CTRL_H * 3  # RGB screenshot of entire controller

# ── Shared memory ────────────────────────────────────────────────────
_shared = None      # multiprocessing.Array('d', 10)
_frame_buf = None   # multiprocessing.RawArray('B', _FRAME_NBYTES)  — onboard camera
_frame_seq = None   # multiprocessing.Value('i')
_panel_buf = None   # multiprocessing.RawArray('B', _PANEL_NBYTES) — full controller screenshot
_panel_seq = None   # multiprocessing.Value('i')
_process = None


def _ensure_shared():
    global _shared, _frame_buf, _frame_seq, _panel_buf, _panel_seq
    if _shared is None:
        _shared = multiprocessing.Array(ctypes.c_double, 10, lock=True)
        for i in range(10):
            _shared[i] = 0.0
    if _frame_buf is None:
        _frame_buf = multiprocessing.RawArray(ctypes.c_uint8, _FRAME_NBYTES)
    if _frame_seq is None:
        _frame_seq = multiprocessing.Value(ctypes.c_int, 0, lock=True)
    if _panel_buf is None:
        _panel_buf = multiprocessing.RawArray(ctypes.c_uint8, _PANEL_NBYTES)
    if _panel_seq is None:
        _panel_seq = multiprocessing.Value(ctypes.c_int, 0, lock=True)


def get_joystick_state():
    """Return a snapshot of the joystick axes (process-safe via shared memory)."""
    if _shared is None:
        return {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0,
                "active": False, "cam_tilt": 0.0}
    try:
        with _shared.get_lock():
            return {
                "surge":    _shared[0],
                "sway":     _shared[1],
                "heave":    _shared[2],
                "yaw":      _shared[3],
                "active":   _shared[4] > 0.5,
                "cam_tilt": _shared[5],
            }
    except Exception:
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


def is_recording():
    """Return True if the panel's REC button is active."""
    if _shared is None:
        return False
    try:
        with _shared.get_lock():
            return _shared[8] > 0.5
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
        return 0, None


# ── Tkinter GUI (runs inside child process) ──────────────────────────
def _panel_main(shared_arr, frame_buf, frame_seq, panel_buf, panel_seq):
    """Entry point for the child process — minimal controller panel."""
    import tkinter as tk

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

    # ── Dimensions ────────────────────────────────────────────────
    CTRL_W = 720
    CTRL_H = 480

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
                roll_r  = shared_arr[6]
                pitch_r = shared_arr[7]
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

    def _heave_update():
        """Write combined heave to shared memory: +1 up, -1 down, 0 off."""
        if _heave_state["up"] and not _heave_state["dn"]:
            _set(2, 1.0)
        elif _heave_state["dn"] and not _heave_state["up"]:
            _set(2, -1.0)
        else:
            _set(2, 0.0)

    def _in_rect(mx, my, x1, y1, x2, y2):
        return x1 <= mx <= x2 and y1 <= my <= y2

    def _heave_press(event):
        mx, my = event.x, event.y
        ux1 = _HEAVE_CX - _HEAVE_BTN_W // 2
        ux2 = _HEAVE_CX + _HEAVE_BTN_W // 2
        if _in_rect(mx, my, ux1, _hbtn_up_y1, ux2, _hbtn_up_y2):
            _heave_state["up"] = True
            canvas.itemconfig(_hbtn_up, fill=COL_BTN_RED_PRESS)
            _heave_update()
            return True
        if _in_rect(mx, my, ux1, _hbtn_dn_y1, ux2, _hbtn_dn_y2):
            _heave_state["dn"] = True
            canvas.itemconfig(_hbtn_dn, fill=COL_BTN_RED_PRESS)
            _heave_update()
            return True
        return False

    def _heave_release(event):
        changed = False
        if _heave_state["up"]:
            _heave_state["up"] = False
            canvas.itemconfig(_hbtn_up, fill=COL_BTN_RED)
            changed = True
        if _heave_state["dn"]:
            _heave_state["dn"] = False
            canvas.itemconfig(_hbtn_dn, fill=COL_BTN_RED)
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
            _set(8, 1.0)
            canvas.itemconfig(_rec_btn, fill=COL_REC_ACTIVE, outline=COL_REC_ACT_OUT)
            canvas.itemconfig(_rec_btn_txt, text="⏹ STOP", fill=COL_REC_ACT_LBL)
            _rec_flash()
        else:
            _set(8, 0.0)
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
            _set(0, ay)       # surge = up/down
            _set(3, ax)       # yaw   = left/right (positive ax = drag right = yaw right)
        elif dr < JOY_R + 20:
            _drag["active"] = "right"
            ax, ay = _move_knob(_rknob, _rknob_dot, JOY_R_CX, JOY_R_CY, mx, my, JOY_R)
            _set(3, _shared_yaw_from_left[0] + ax)  # right stick X adds to yaw
            _set(5, ay)       # cam_tilt = up/down

    def _on_drag(event):
        mx, my = event.x, event.y
        if _drag["active"] == "left":
            ax, ay = _move_knob(_lknob, _lknob_dot, JOY_L_CX, JOY_L_CY, mx, my, JOY_R)
            _set(0, ay)       # surge
            _set(3, ax)       # yaw from left stick (positive ax = drag right = yaw right)
            _shared_yaw_from_left[0] = ax
        elif _drag["active"] == "right":
            ax, ay = _move_knob(_rknob, _rknob_dot, JOY_R_CX, JOY_R_CY, mx, my, JOY_R)
            _set(3, _shared_yaw_from_left[0] + ax)  # right stick X adds to yaw
            _set(5, ay)       # cam_tilt

    def _on_release(event):
        if _drag["active"] == "heave":
            _heave_release(event)
        elif _drag["active"] == "left":
            _snap_knob(_lknob, _lknob_dot, JOY_L_CX, JOY_L_CY)
            _set(0, 0.0)
            _shared_yaw_from_left[0] = 0.0
            # Preserve right-stick yaw contribution if right stick is also active
            _set(3, 0.0)
        elif _drag["active"] == "right":
            _snap_knob(_rknob, _rknob_dot, JOY_R_CX, JOY_R_CY)
            _set(3, _shared_yaw_from_left[0])  # Restore to just left-stick yaw
            _set(5, 0.0)
        _drag["active"] = None

    canvas.bind("<ButtonPress-1>", _on_press)
    canvas.bind("<B1-Motion>", _on_drag)
    canvas.bind("<ButtonRelease-1>", _on_release)

    _set(4, 1.0)

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
        if not _has_imagegrab:
            return
        try:
            # Check if recording is active
            with shared_arr.get_lock():
                rec_on = shared_arr[8] > 0.5
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
            pass

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
        # Capture panel screenshot for recording (if active)
        _capture_panel()
        root.after(_CAM_POLL_MS, _update_camera)

    _update_camera()

    # ── Close handler ─────────────────────────────────────────────
    def _on_close():
        _set(4, 0.0)
        _set(8, 0.0)   # stop recording
        for i in range(6):
            _set(i, 0.0)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()
    _set(4, 0.0)
    _set(8, 0.0)   # stop recording
    for i in range(6):
        _set(i, 0.0)


# ── Public API ────────────────────────────────────────────────────────
def start_joystick_panel():
    """Launch the joystick panel in a child process."""
    global _process, _shared
    _ensure_shared()
    if _process is not None and _process.is_alive():
        return
    _process = multiprocessing.Process(
        target=_panel_main,
        args=(_shared, _frame_buf, _frame_seq, _panel_buf, _panel_seq),
        daemon=True, name="JoystickPanel")
    _process.start()
    time.sleep(0.25)


def stop_joystick_panel():
    """Request the panel process to stop (idempotent)."""
    global _process, _shared
    if _shared is not None:
        try:
            with _shared.get_lock():
                _shared[4] = 0.0
                _shared[8] = 0.0   # stop recording
                for i in range(6):
                    _shared[i] = 0.0
        except Exception:
            pass
    if _process is not None:
        try:
            _process.terminate()
            _process.join(timeout=1.0)
        except Exception:
            pass
        _process = None


# ── Thruster mixer ───────────────────────────────────────────────────
def clamp_val(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def mix_joystick_to_thruster_cmds(state, n_thrusters):
    """
    Convert joystick axes to per-thruster ON/OFF commands.

    IMPORTANT: Real thrusters are binary — they can only be ON (full power)
    or OFF.  There is no variable-speed PWM.  The mixer therefore outputs
    only -1, 0, or +1 for each thruster.

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

    Each raw value is then snapped to -1, 0, or +1:
      |raw| < threshold  → 0 (OFF)
      raw > 0            → +1 (ON forward)
      raw < 0            → -1 (ON reverse)

    Sign convention (verified from geometry):
      T1 cmd=+1 → yaw torque +1.14 N·m (CCW = yaw LEFT)
      T2 cmd=+1 → yaw torque -1.14 N·m (CW  = yaw RIGHT)
    So to yaw RIGHT (CW, negative τz): decrease T1, increase T2.

    Dead zone at 15% overall stick magnitude.
    T3 (heave) is keyboard-only — always 0 from joystick.
    """
    surge = state.get("surge", 0.0)
    yaw   = state.get("yaw", 0.0)

    DEAD = 0.15
    mag = math.sqrt(surge * surge + yaw * yaw)

    cmds = [0.0] * n_thrusters
    if mag < DEAD:
        return cmds

    # Compute raw proportional values, then snap to binary.
    # All horizontal thrusters contribute to surge,
    # T1/T2 differential creates yaw torque.
    raw_t1 = surge - yaw   # yaw right → less T1 (less CCW torque)
    raw_t2 = surge + yaw   # yaw right → more T2 (more CW torque)
    raw_t4 = surge          # pure forward/reverse

    # Snap to binary: -1, 0, or +1
    SNAP_THRESHOLD = 0.15   # below this magnitude → OFF
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
    # T3 (vertical) left as 0 — keyboard only
    if n_thrusters >= 4:
        cmds[3] = t4
    return cmds

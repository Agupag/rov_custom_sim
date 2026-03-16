# Project Context

This document is a compact engineering briefing for future maintainers and AI agents working in this workspace.

## Source Of Truth

The primary implementation is `rov_sim.py`.

- It is a single-file application, not a thin wrapper around a package.
- Many helpers used by tests and tools live inside it and should be preserved unless there is a strong reason to change public behavior.

The secondary runtime file is `joystick_panel.py`.

- It owns the Tkinter controller panel.
- It also owns the shared-memory contract and the joystick mixer.

## Main Runtime Structure

`rov_sim.py` contains:

- file logging setup
- top-level physics and rendering constants
- mesh and GLTF helpers
- buoyancy, ballast, righting, and drag application helpers
- obstacle water-force handling
- ROV creation and optional visual marker helpers
- thruster indicator creation and updates
- the main PyBullet GUI loop

The file is large by design, so changes should be incremental and well-scoped.

## Inter-Process Data Flow

The controller panel runs in a separate process and communicates through shared memory.

Shared memory layout in `joystick_panel.py`:

- `[0] surge`
- `[1] sway`
- `[2] heave`
- `[3] yaw`
- `[4] active`
- `[5] cam_tilt`
- `[6] roll_rad`
- `[7] pitch_rad`
- `[8] rec_flag`
- `[9] depth_m`
- `[10] heading_deg`
- `[11] speed_mps`
- `[12] thrust_level`
- `[13] depth_hold_active`
- `[14] heading_hold_active`
- `[15] rec_status`
- `[16] control_mode`
- `[17] set_thrust_level`
- `[18] set_proportional_mode`
- `[19] set_depth_hold`
- `[20] set_heading_hold`
- `[21] set_cam_follow`
- `[22] set_cam_chase`
- `[23] set_topdown`
- `[24] set_show_force_vectors`
- `[25] set_thruster_failure`
- `[26] set_emergency_surface`
- `[27] cmd_reset_rov`
- `[28] set_trail_enabled`
- `[29-31] reserved`

There are also raw RGB buffers for the onboard camera frame and full panel screenshot.

## Control Semantics

The current joystick mixer defaults to binary output and also supports an optional proportional mode.

- Default mode returns `-1`, `0`, or `+1` per thruster.
- Optional proportional mode outputs continuous values in `[-1, +1]` and is used for diagnostics and tuning workflows.
- `T1 = surge - yaw`
- `T2 = surge + yaw`
- `T4 = surge`
- `T3` is controlled separately through the heave path rather than the joystick mixer output.

Smooth behavior comes from spool dynamics in `rov_sim.py`, not from analog PWM commands alone.

The reverse sign-change cooldown is enforced in the simulator loop, not inside the mixer.

## Physics Conventions

This project intentionally uses a marine-vehicle style model.

- buoyancy and ballast are modeled separately
- righting torque is explicit
- linear and quadratic drag are applied in the body frame
- added mass and added inertia are represented as diagonal terms
- added-mass Coriolis coupling is modeled explicitly
- forces and torques are rotated into world coordinates before PyBullet application

When changing the physics, preserve frame consistency. A common failure mode is mixing body-frame and world-frame quantities in the same calculation.

## Thruster Geometry

The preferred geometry source is `Assembly 1.gltf`.

- `detect_thrusters_from_gltf(...)` should be trusted before the fallback `THRUSTERS` constant.
- Avoid hardcoding replacement thrust vectors unless GLTF detection is known to be broken.

The simulator also uses the GLTF when possible to locate the onboard camera pose.

## Validation Workflow

The diagnostics are the intended validation layer.

- `test_extended_diagnostics.py` is the assist-mode and GLTF frame validation suite and writes a baseline JSON report.
- `test_sim_diagnostic.py` is useful when changing signs, axes, or thruster behavior.
- `test_physics_realistic.py` is useful when tuning drag, added mass, buoyancy, or spool constants.
- `test_mixer_analysis.py` is useful when reasoning about the geometry and force decomposition.
- `test_joystick.py` and `test_joystick_full.py` are useful when touching shared memory or cooldown logic.
- `test_thruster_logic.py` is the fast check for persistent reverse behavior.
- `tools/run_validation_and_calibration.py` runs the full diagnostics plus calibration artifact regeneration pipeline in one command.

These scripts are behavior-oriented. They are not all meant to be run under pytest as strict automated unit tests.

## Useful Tuning Areas

The most important tuning parameters are grouped near the top of `rov_sim.py`.

- hydrodynamics: `LIN_DRAG_BODY`, `LIN_DRAG_ANG`, `CD`, `AREA`, `QUAD_DRAG_ANG`
- hydrostatics: `BUOYANCY_SCALE`, `COB_OFFSET_BODY`, `BALLAST_OFFSET_BODY`, `BALLAST_SCALE`
- stability: `RIGHTING_K_RP`, `RIGHTING_KD_RP`
- inertia: `ADDED_MASS_BODY`, `ADDED_INERTIA_BODY`, `CORIOLIS_SCALE`
- propulsion: `MAX_THRUST_H`, `MAX_THRUST_V`, `THRUSTER_TAU_UP`, `THRUSTER_TAU_DN`, `BACKWARDS_THRUST_SCALE`, `THRUSTER_SPEED_LOSS_COEF`

Prefer constants-first iteration before structural rewrites.

## Trust Ranking

Most trustworthy:

- `rov_sim.py`
- `joystick_panel.py`
- `test_sim_diagnostic.py`
- `test_physics_realistic.py`
- `test_mixer_analysis.py`
- `test_joystick.py`

Useful but secondary:

- `physics_analyzer.py`
- `physics_auto_optimizer.py`
- `tools/tune_added_mass_and_thruster_loss.py`

Not authoritative for current behavior:

- `rov_sim_backup.py`
- generated logs
- virtualenv directories

## Workspace Notes

- `README.md` is intended for human onboarding and should stay aligned with the current files in the workspace.
- `README_WINDOWS.md` covers the standalone Windows build path.
- `build_windows_exe.bat` and `rov_sim.spec` are part of a maintained packaging flow.
- `0_VRDemoSettings.txt` appears vestigial.

## Safe Change Strategy

When making changes:

1. Confirm whether the behavior belongs in the joystick panel, the simulator loop, or a helper used by both.
2. Preserve GLTF-driven transforms when possible.
3. Keep body-frame versus world-frame handling explicit.
4. Use at least one relevant diagnostic script after behavior changes.
5. Avoid broad refactors of `rov_sim.py` unless you are also updating the diagnostic assumptions.
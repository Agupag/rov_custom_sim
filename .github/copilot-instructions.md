# Copilot instructions for `rov_custom_sim`

## Big picture architecture
- Core simulator is `rov_sim.py` (single-file app, ~3k lines): PyBullet world setup, ROV construction, hydrodynamics, thruster application, camera/HUD, logging, and keyboard control all live here.
- Input/UI is split into a separate process in `joystick_panel.py` (Tkinter + shared memory) to avoid macOS main-thread conflicts between Tk and PyBullet GUI.
- Data flow: `joystick_panel` writes shared joystick axes (`surge/yaw/heave/cam_tilt`) -> `rov_sim.py` reads and mixes to thruster commands -> physics step applies buoyancy/ballast/righting/drag/thrust -> optional camera frames pushed back to panel.
- Thruster geometry is usually discovered from `Assembly 1.gltf` via `detect_thrusters_from_gltf(...)`; do not hardcode thrust vectors unless auto-detection fails.

## Physics model conventions (project-specific)
- This repo intentionally uses a marine-vehicle style model (Fossen-inspired): added mass (`ADDED_MASS_BODY`), added inertia, Coriolis coupling, linear + quadratic damping, and hydrostatic righting.
- Keep forces in consistent frames: many calculations are in **body frame**, then rotated to world before `p.applyExternalForce(...)`.
- Buoyancy and ballast are separated concepts (`COB_OFFSET_BODY`, `BALLAST_OFFSET_BODY`) and both matter for stability behavior.
- Thruster command semantics are signed and persistent:
  - `+1/-1/0` command levels
  - reverse mode can be toggled and remembered per thruster (see `test_thruster_logic.py`).
- Thruster spool dynamics are first-order with asymmetric constants (`THRUSTER_TAU_UP` vs `THRUSTER_TAU_DN`); preserve this when adjusting control behavior.

## Critical workflows
- Main run path: execute `rov_sim.py` directly (PyBullet GUI mode).
- Fast physics validation pattern (used in repo tests): run PyBullet in `DIRECT`, disable expensive visuals, and call the same sim helpers as production (`build_rov`, `apply_drag`, `apply_ballast`, `apply_righting_torque`).
- Most meaningful diagnostics are script-style tests that print measured behavior, not strict unit assertions:
  - `test_physics_realistic.py` (accel, terminal speed, stopping distance, yaw/vertical response)
  - `test_sim_diagnostic.py` (mixer + full physics sign-chain verification)
  - `test_mixer_analysis.py` (thruster geometry decomposition and mixer effectiveness)

## Coding patterns to follow in this repo
- Prefer reusing `rov_sim` module functions/constants in tests and tools instead of duplicating physics equations.
- When changing controls/mixing, verify both:
  - pure mixer output (`mix_joystick_to_thruster_cmds` in `joystick_panel.py`)
  - resulting physical direction/torque in `DIRECT` simulation.
- Keep headless test setup consistent with existing scripts:
  - `SLEEP_REALTIME=False`, HUD/markers/camera preview off.
- Preserve logging behavior in `rov_sim.py`: custom `print(...)` mirrors to timestamped `rov_sim_log_*.txt` and is relied on for diagnostics.

## Integration points / dependencies
- External runtime deps are `pybullet`; `opencv-python` and `numpy` are optional in parts of the app.
- Asset coupling is important:
  - `Assembly 1.obj` / `Assembly 1.gltf` drive visual mesh and thruster/camera transform extraction.
- Windows packaging path exists (`rov_sim.spec`, `build_windows_exe.bat`, `README_WINDOWS.md`) and should stay compatible with PyInstaller behavior (`sys.frozen`, `sys._MEIPASS`).

## Safe edit guidance
- Avoid broad refactors of `rov_sim.py` without preserving public helpers used by tests (`build_rov`, `detect_thrusters_from_gltf`, drag/righting helpers).
- For behavior changes, update/extend at least one diagnostic script to demonstrate measured impact (speed, yaw torque, stopping, etc.).
- Keep constants grouped in config blocks at top of `rov_sim.py`; this project tunes behavior by constants-first iteration.
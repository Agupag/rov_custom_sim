# ROV Custom Simulator

This workspace contains a PyBullet-based underwater ROV simulator with a separate Tkinter controller panel, GLTF-driven thruster detection, and several diagnostic scripts for validating physics and control behavior.

## What Is Here

- `rov_sim.py`: main simulator and primary entrypoint
- `joystick_panel.py`: controller UI process and joystick mixer
- `Assembly 1.obj` / `Assembly 1.gltf` / `Assembly 1.mtl`: mesh and transform assets used by the simulator
- `test_*.py`: behavior-focused diagnostics for controls, thruster logic, and physics
- `physics_analyzer.py`: post-run log parser for detailed physics logs
- `physics_auto_optimizer.py`: heuristic analysis helper for log-driven tuning
- `tools/tune_added_mass_and_thruster_loss.py`: offline tuning helper that inspects logs and current constants
- `README_WINDOWS.md`: Windows build and distribution notes
- `rov_sim.spec` / `build_windows_exe.bat`: PyInstaller packaging path for Windows

## Architecture

The runtime is centered on `rov_sim.py`.

- It builds the PyBullet world, loads the ROV mesh, detects thrusters from the GLTF, runs the hydrodynamics, applies thrust, renders the scene, logs telemetry, and handles keyboard control.
- It uses a marine-vehicle style model with buoyancy, ballast, righting torque, linear and quadratic drag, added mass, added inertia, and added-mass Coriolis coupling.
- Most force calculations are performed in the body frame and then rotated into the world frame before being applied to PyBullet.

The controller UI is isolated in `joystick_panel.py`.

- On macOS this separation matters because Tkinter and the PyBullet GUI both prefer main-thread ownership.
- Shared memory carries joystick axes and camera frames between the two processes.
- The joystick mixer is binary, not proportional: it outputs `-1`, `0`, or `+1` per thruster, and the perceived smoothness comes from thruster spool dynamics in `rov_sim.py`.

## Control Flow

1. The joystick panel writes `surge`, `yaw`, `heave`, and `cam_tilt` into shared memory.
2. `rov_sim.py` reads those values once per step.
3. `joystick_panel.mix_joystick_to_thruster_cmds(...)` maps joystick intent to binary thruster commands.
4. `rov_sim.py` applies per-thruster sign-change cooldown logic and first-order ramping.
5. The sim computes buoyancy, ballast, drag, added-mass effects, and thrust, then steps PyBullet.
6. Optional camera frames are written back to the controller panel for preview and recording.

## Thrusters And Physics

Important project conventions:

- Thruster commands are signed and persistent: `+1` forward, `-1` reverse, `0` off.
- Reverse thrust is intentionally weaker than forward thrust.
- Thruster spool-up and spool-down use different time constants.
- Thruster geometry should come from `Assembly 1.gltf` when available. Avoid hardcoding directions unless GLTF detection fails.
- Buoyancy and ballast are separate concepts and are both important for stability behavior.

Important tuning constants live near the top of `rov_sim.py`, including:

- `MASS`
- `BUOYANCY_SCALE`
- `COB_OFFSET_BODY`
- `BALLAST_OFFSET_BODY`
- `BALLAST_SCALE`
- `LIN_DRAG_BODY`
- `LIN_DRAG_ANG`
- `CD`
- `AREA`
- `ADDED_MASS_BODY`
- `ADDED_INERTIA_BODY`
- `RIGHTING_K_RP`
- `RIGHTING_KD_RP`
- `MAX_THRUST_H`
- `MAX_THRUST_V`
- `THRUSTER_TAU_UP`
- `THRUSTER_TAU_DN`
- `BACKWARDS_THRUST_SCALE`
- `THRUSTER_SPEED_LOSS_COEF`

## Running The Simulator

Python 3.12 is the safest choice for PyBullet compatibility.

Install dependencies in your environment:

```bash
pip install pybullet numpy opencv-python Pillow
```

Run the simulator:

```bash
python3 rov_sim.py
```

There is also a convenience launcher for a conda environment named `rov_conda`:

```bash
./run_sim_conda.sh
```

## Diagnostics And Tests

These scripts are the best guide to intended behavior.

- `test_physics_realistic.py`: measures terminal speed, acceleration, stopping distance, yaw response, and vertical response
- `test_sim_diagnostic.py`: verifies mixer output and end-to-end sign chain through real physics
- `test_mixer_analysis.py`: decomposes thruster geometry and force or torque contributions
- `test_physics_diag.py`: runs a timed headless diagnostic and writes `diag_physics.log`
- `test_joystick.py`: checks mixer mapping, shared memory, cooldown logic, and integration behavior
- `test_joystick_full.py`: runs a multi-phase joystick-driven headless scenario
- `test_thruster_logic.py`: validates the reverse and on/off state machine without PyBullet

These are mostly script-style diagnostics rather than strict assertion-heavy unit tests.

## Logs

Every simulator run writes a timestamped log file in the workspace:

- `rov_sim_log_YYYYMMDD_HHMMSS.txt`

If `LOG_PHYSICS_DETAILED` is enabled in `rov_sim.py`, the log includes a `# DETAILED_PHYSICS_CSV` section that can be parsed by the analysis helpers.

## Windows Packaging

For Windows standalone builds:

- Use `build_windows_exe.bat`
- PyInstaller configuration lives in `rov_sim.spec`
- Full build instructions are in `README_WINDOWS.md`

## Files To Treat Carefully

- `rov_sim.py` is the canonical runtime implementation.
- `rov_sim_backup.py` appears to be an older backup and should not be treated as the source of truth.
- `rov_sim_log_*.txt`, `diag_physics.log`, `__pycache__/`, and virtualenv directories are generated artifacts or environment state.

## Recommended Editing Approach

- Prefer changing constants and validating behavior with the diagnostic scripts before attempting broad refactors.
- Preserve helper functions used by the diagnostics, especially `build_rov`, `detect_thrusters_from_gltf`, and the hydrodynamic helpers.
- Keep frame conventions consistent: body-frame calculations first, world-frame application second.
- When changing controls or thruster logic, validate both the mixer behavior and the resulting physical motion.

For a deeper architecture and maintenance briefing, see `PROJECT_CONTEXT.md`.

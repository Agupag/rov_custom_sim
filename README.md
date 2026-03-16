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

- It builds the PyBullet world, loads the ROV mesh, detects thrusters from the GLTF, runs the hydrodynamics, applies thrust, renders the scene, logs telemetry, and applies runtime settings from the controller panel.
- It uses a marine-vehicle style model with buoyancy, ballast, righting torque, linear and quadratic drag, added mass, added inertia, and added-mass Coriolis coupling.
- Most force calculations are performed in the body frame and then rotated into the world frame before being applied to PyBullet.

The controller UI is isolated in `joystick_panel.py`.

- On macOS this separation matters because Tkinter and the PyBullet GUI both prefer main-thread ownership.
- Shared memory carries joystick axes, telemetry, recording status, control mode, and camera frames between the two processes.
- The joystick mixer is binary by default: it outputs `-1`, `0`, or `+1` per thruster, and the perceived smoothness comes from thruster spool dynamics in `rov_sim.py`.
- A proportional path also exists for diagnostics and fine control experiments, but binary mode remains the default operator workflow.

### Controller Panel UX

- On-screen legend now shows quick controls for sticks, heave, recording, and close.
- Status chips in the panel indicate control mode, assist state, and recording health with color-coded feedback.
- Bottom telemetry remains depth, heading, speed, and thrust, and now pairs with clearer mode and recording labels.
- The `SETTINGS` button opens a dedicated runtime settings window.
- Simulator settings are controlled through this Tkinter settings window (not keyboard keys in the simulator).

## Control Flow

1. The joystick panel writes joystick axes and settings values into shared memory.
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

- `test_extended_diagnostics.py`: validates depth hold, heading hold, GLTF thruster frame checks, and exports `extended_diagnostics_results.json`
- `test_physics_realistic.py`: measures terminal speed, acceleration, stopping distance, yaw response, vertical response, and loop timing metrics
- `test_sim_diagnostic.py`: verifies mixer output and end-to-end sign chain through real physics
- `test_mixer_analysis.py`: decomposes thruster geometry and force or torque contributions
- `test_physics_diag.py`: runs a timed headless diagnostic and writes `diag_physics.log`
- `test_joystick.py`: checks mixer mapping, shared memory, cooldown logic, and integration behavior
- `test_joystick_full.py`: runs a multi-phase joystick-driven headless scenario
- `test_thruster_logic.py`: validates the reverse and on or off state machine without PyBullet

Calibration tooling:

- `tools/run_sensitivity_sweep.py`: runs one-factor-at-a-time physics sensitivity sweeps and writes `tools/sensitivity_sweep_results.json` and `.csv`
- `tools/analyze_sensitivity_recommendation.py`: scores sweep scenarios and writes recommendation JSON and Markdown outputs
- `tools/run_validation_and_calibration.py`: one-command pipeline that runs diagnostics and calibration tooling and writes `tools/validation_pipeline_summary.json`

These are mostly script-style diagnostics rather than strict assertion-heavy unit tests.

## Verification Harness

The repository now includes a structured debug harness under `debug/` for evidence-first verification.

- `python -m debug.debug_full_system`: runs the full suite and writes a run folder under `debug_artifacts/`
- `python -m debug.debug_startup_and_config`: startup, asset, config, and dependency preflight
- `python -m debug.debug_thruster_geometry`: GLTF thruster vector/count/symmetry/yaw-sign checks
- `python -m debug.debug_control_path`: shared-memory to mixer to cooldown to thrust trace capture
- `python -m debug.debug_physics_sanity`: settle, surge, yaw, and heave qualitative sanity checks
- `python -m debug.debug_physics_environment_stress`: adversarial command stress test across environment presets (NaN/instability bounds)
- `python -m debug.debug_camera_recording_pipeline`: camera pose source, frame freshness, recording prerequisites, and recording file integrity probe
- `python -m debug.debug_ui_truthfulness`: shared-memory/UI status mapping plus joystick, recording-status, and panel-frame roundtrip checks
- `python -m debug.debug_ui_settings_contract`: settings menu slot wiring, range contracts, reset pulse semantics, and feedback-slot coverage
- `python -m debug.debug_runtime_events_integrity`: validates env-gated JSONL runtime event emission and schema
- `python -m debug.debug_recording_event_file_correlation`: validates recording start/stop events correlate to a real non-empty recording artifact path
- `python -m debug.debug_runtime_consistency`: doc/runtime mismatch and stale-file risk checks

Each module writes `results.json` and `results.md` in its own artifact folder, plus raw traces when relevant.

Optional runtime event instrumentation can be enabled while running the simulator:

- `ROV_DEBUG_EVENTS=1` enables JSONL event logs for simulator and panel lifecycle/state transitions
- `ROV_DEBUG_EVENTS_FILE=/path/to/runtime_events.jsonl` overrides the output path

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

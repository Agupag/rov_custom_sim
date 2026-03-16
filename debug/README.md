# Debug and Verification Harness

This folder contains evidence-first verification scripts for the ROV simulator.

## Goals

- Verify runtime behavior with artifacts instead of assumptions.
- Capture machine-readable pass/fail evidence for startup, controls, thrusters, and physics.
- Make fallback paths and hidden state transitions observable.

## Modules

- `debug_full_system.py`: orchestrates all modules and writes aggregate summaries.
- `debug_startup_and_config.py`: validates required files, configuration inventory, and dependency flags.
- `debug_thruster_geometry.py`: validates GLTF-derived thruster geometry and yaw sign chain.
- `debug_control_path.py`: traces shared-memory input through mixer, cooldown logic, and thrust application.
- `debug_physics_sanity.py`: checks settle drift and qualitative surge/yaw/heave responses.
- `debug_physics_environment_stress.py`: stress-tests physics and environment presets for finite-state stability and bounded kinematics.
- `debug_camera_recording_pipeline.py`: checks camera pose source, frame freshness, recording prerequisites, and recording file integrity probe.
- `debug_ui_truthfulness.py`: checks shared-memory/UI label truthfulness plus joystick, recording-status, and panel-frame signal paths.
- `debug_ui_settings_contract.py`: checks settings slot wiring, range contracts, reset semantics, and simulator feedback path coverage.
- `debug_runtime_events_integrity.py`: validates runtime JSONL event emission, source coverage, and schema integrity.
- `debug_recording_event_file_correlation.py`: correlates recording lifecycle events with a tangible recording file artifact path.
- `debug_runtime_consistency.py`: flags doc/runtime mismatch risks and stale legacy files.
- `common.py`: shared artifact and result helpers.
- `scenarios.py`: deterministic command phase definitions.
- `runtime_events.py`: env-gated JSONL runtime event logger.

## Run

From the workspace root:

```bash
python -m debug.debug_full_system
```

Artifacts are written to:

- `debug_artifacts/<run_id>/summary.json`
- `debug_artifacts/<run_id>/summary.md`
- `debug_artifacts/<run_id>/<module>/results.json`
- `debug_artifacts/<run_id>/<module>/results.md`

Optional instrumentation for live simulator runs:

```bash
ROV_DEBUG_EVENTS=1 /opt/homebrew/anaconda3/envs/rov_conda/bin/python rov_sim.py
```

To force a specific event-log file path:

```bash
ROV_DEBUG_EVENTS=1 ROV_DEBUG_EVENTS_FILE=debug_artifacts/runtime/runtime_events.jsonl /opt/homebrew/anaconda3/envs/rov_conda/bin/python rov_sim.py
```

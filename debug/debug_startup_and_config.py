#!/usr/bin/env python3
"""Startup and configuration verification for the ROV simulator."""

from __future__ import annotations

import os
from pathlib import Path

from debug.common import (
    CheckResult,
    capture_environment_snapshot,
    console,
    emit_module_results,
    init_context,
    parse_common_args,
    write_json,
)


def main() -> int:
    args = parse_common_args("Verify startup, config, assets, and dependency assumptions")
    ctx = init_context("startup", args)

    import rov_sim

    results: list[CheckResult] = []

    env = capture_environment_snapshot()
    write_json(ctx.module_dir / "environment.json", env)

    required_runtime = [
        "rov_sim.py",
        "joystick_panel.py",
        "sim_shared.py",
        "Assembly 1.obj",
        "Assembly 1.gltf",
    ]
    missing_runtime = [p for p in required_runtime if not Path(rov_sim.DATA_DIR, p).exists()]
    status = "pass" if not missing_runtime else "fail"
    results.append(
        CheckResult(
            name="required_runtime_files",
            status=status,
            summary="Checked core runtime and primary asset presence.",
            evidence={"required": required_runtime, "missing": missing_runtime},
        )
    )

    cfgs = {}
    asset_warnings: list[str] = []
    for cfg_name, cfg in rov_sim.THRUSTER_CONFIGS.items():
        obj_ok = Path(cfg["obj"]).exists()
        gltf_ok = Path(cfg["gltf"]).exists()
        cfgs[cfg_name] = {
            "obj": cfg["obj"],
            "gltf": cfg["gltf"],
            "obj_exists": obj_ok,
            "gltf_exists": gltf_ok,
        }
        if not obj_ok or not gltf_ok:
            asset_warnings.append(f"{cfg_name}: missing obj or gltf")

    config_ok = bool(cfgs) and not asset_warnings
    results.append(
        CheckResult(
            name="thruster_config_inventory",
            status="pass" if config_ok else "warn",
            summary="Enumerated discovered THRUSTER_CONFIGS and validated file pairs.",
            evidence={
                "active_config_name": rov_sim.ACTIVE_CONFIG_NAME,
                "active_obj": rov_sim.OBJ_FILE,
                "active_gltf": rov_sim.GLTF_FILE,
                "config_count": len(cfgs),
                "configs": cfgs,
            },
            warnings=asset_warnings,
        )
    )

    fallback_reasons = []
    if not Path(rov_sim.GLTF_FILE).exists():
        fallback_reasons.append("active GLTF file missing; auto-detect cannot run")
    if not Path(rov_sim.OBJ_FILE).exists():
        fallback_reasons.append("active OBJ file missing; build_rov will fail")

    auto_detect_status = "pass" if rov_sim.AUTO_DETECT_THRUSTERS else "warn"
    auto_detect_summary = "AUTO_DETECT_THRUSTERS enabled" if rov_sim.AUTO_DETECT_THRUSTERS else "AUTO_DETECT_THRUSTERS disabled"
    results.append(
        CheckResult(
            name="auto_detect_policy",
            status=auto_detect_status,
            summary=auto_detect_summary,
            evidence={
                "AUTO_DETECT_THRUSTERS": rov_sim.AUTO_DETECT_THRUSTERS,
                "fallback_thruster_count": len(rov_sim.THRUSTERS),
            },
            warnings=fallback_reasons,
        )
    )

    optional_deps = {
        "numpy": bool(getattr(rov_sim, "HAS_NUMPY", False)),
        "opencv": bool(getattr(rov_sim, "HAS_CV2", False)),
        "pillow": bool(getattr(rov_sim, "HAS_PIL", False)),
    }
    dep_warnings = []
    if not optional_deps["numpy"]:
        dep_warnings.append("numpy missing; camera frame conversion paths will degrade")
    if not optional_deps["opencv"]:
        dep_warnings.append("opencv-python missing; recording and fallback preview paths are limited")
    if not optional_deps["pillow"]:
        dep_warnings.append("Pillow missing; some panel capture paths may be unavailable")
    results.append(
        CheckResult(
            name="optional_dependencies",
            status="pass" if not dep_warnings else "warn",
            summary="Inspected optional dependency flags used by camera and recording paths.",
            evidence=optional_deps,
            warnings=dep_warnings,
        )
    )

    packaging_assets = [
        "Assembly 1.obj",
        "Assembly 1.gltf",
        "v1.obj",
        "v1.gltf",
        "v2.obj",
        "v2.gltf",
        "v3.obj",
        "v3.gltf",
    ]
    missing_packaging = [name for name in packaging_assets if not Path(rov_sim.DATA_DIR, name).exists()]
    results.append(
        CheckResult(
            name="packaging_asset_preflight",
            status="pass" if not missing_packaging else "warn",
            summary="Validated expected assets used by PyInstaller spec are present in workspace.",
            evidence={"expected_assets": packaging_assets, "missing": missing_packaging},
        )
    )

    payload = emit_module_results(ctx, "startup", results, extras={"environment": env})
    console(f"[startup] wrote: {ctx.module_dir / 'results.json'}", quiet=args.quiet)
    console(f"[startup] checks: {payload['counts']}", quiet=args.quiet)

    return 1 if payload["counts"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

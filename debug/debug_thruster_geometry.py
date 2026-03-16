#!/usr/bin/env python3
"""Thruster geometry and direction verification against GLTF/runtime expectations."""

from __future__ import annotations

import math
from pathlib import Path

from debug.common import CheckResult, console, emit_module_results, init_context, parse_common_args


def _vmag(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vcross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _score_symmetry(h_thr: list[dict]) -> dict:
    if len(h_thr) < 2:
        return {"ok": False, "reason": "insufficient horizontal thrusters"}

    left = [t for t in h_thr if t["pos"][0] < 0]
    right = [t for t in h_thr if t["pos"][0] > 0]
    if not left or not right:
        return {"ok": False, "reason": "missing left/right split"}

    lx = sum(abs(t["pos"][0]) for t in left) / len(left)
    rx = sum(abs(t["pos"][0]) for t in right) / len(right)
    gap = abs(lx - rx)
    return {
        "ok": gap < 0.08,
        "left_abs_x_mean": lx,
        "right_abs_x_mean": rx,
        "symmetry_gap": gap,
    }


def main() -> int:
    args = parse_common_args("Verify GLTF-driven thruster geometry and force/torque signs")
    ctx = init_context("thrusters", args)

    import rov_sim

    results: list[CheckResult] = []
    cfg_reports: dict[str, dict] = {}

    for cfg_name, cfg in rov_sim.THRUSTER_CONFIGS.items():
        obj_path = Path(cfg["obj"])
        gltf_path = Path(cfg["gltf"])
        report = {
            "obj": str(obj_path),
            "gltf": str(gltf_path),
            "thrusters": [],
        }

        if not obj_path.exists() or not gltf_path.exists():
            cfg_reports[cfg_name] = report
            results.append(
                CheckResult(
                    name=f"config_{cfg_name}_files",
                    status="fail",
                    summary="Config missing required OBJ/GLTF files.",
                    evidence=report,
                )
            )
            continue

        center, _size = rov_sim.obj_bounds(str(obj_path))
        thrusters = rov_sim.detect_thrusters_from_gltf(str(gltf_path), center)
        report["thruster_count"] = len(thrusters)

        for thr in thrusters:
            d = tuple(float(x) for x in thr["dir"])
            report["thrusters"].append(
                {
                    "name": thr["name"],
                    "kind": thr["kind"],
                    "pos": list(thr["pos"]),
                    "dir": list(d),
                    "dir_norm": _vmag(d),
                }
            )

        cfg_reports[cfg_name] = report

        if len(thrusters) != 4:
            results.append(
                CheckResult(
                    name=f"config_{cfg_name}_thruster_count",
                    status="fail",
                    summary="Expected 4 thrusters from GLTF detection.",
                    evidence={"expected": 4, "actual": len(thrusters)},
                )
            )
            continue

        bad_norm = [t for t in report["thrusters"] if abs(t["dir_norm"] - 1.0) > 1e-3]
        results.append(
            CheckResult(
                name=f"config_{cfg_name}_vector_norms",
                status="pass" if not bad_norm else "fail",
                summary="Checked normalized direction vectors.",
                evidence={"bad_norm_count": len(bad_norm), "bad_norm": bad_norm},
            )
        )

        h_thr = [t for t in thrusters if t["kind"] == "H"]
        v_thr = [t for t in thrusters if t["kind"] == "V"]
        sym = _score_symmetry(h_thr)
        v_ok = bool(v_thr) and all(t["dir"][2] > 0.5 for t in v_thr)

        results.append(
            CheckResult(
                name=f"config_{cfg_name}_layout_sanity",
                status="pass" if sym.get("ok") and v_ok else "warn",
                summary="Checked horizontal symmetry and vertical thruster orientation.",
                evidence={"symmetry": sym, "vertical_ok": v_ok, "vertical_count": len(v_thr)},
            )
        )

        # Force/torque sign expectations for canonical test commands.
        net_tz_yaw_left = 0.0
        net_tz_yaw_right = 0.0
        for idx, t in enumerate(thrusters):
            r = t["pos"]
            d = t["dir"]
            cmd_left = 0.0
            cmd_right = 0.0
            # mirror of binary mixer expectations: yaw<0 => T1 +1, T2 -1
            if idx == 0:
                cmd_left = 1.0
                cmd_right = -1.0
            elif idx == 1:
                cmd_left = -1.0
                cmd_right = 1.0
            f_l = (d[0] * cmd_left, d[1] * cmd_left, d[2] * cmd_left)
            f_r = (d[0] * cmd_right, d[1] * cmd_right, d[2] * cmd_right)
            net_tz_yaw_left += _vcross(r, f_l)[2]
            net_tz_yaw_right += _vcross(r, f_r)[2]

        results.append(
            CheckResult(
                name=f"config_{cfg_name}_yaw_sign_chain",
                status="pass" if net_tz_yaw_left > 0.0 and net_tz_yaw_right < 0.0 else "fail",
                summary="Validated expected yaw torque sign for mixer differential commands.",
                evidence={
                    "net_tz_for_yaw_left_input": net_tz_yaw_left,
                    "net_tz_for_yaw_right_input": net_tz_yaw_right,
                },
            )
        )

    payload = emit_module_results(ctx, "thrusters", results, extras={"configs": cfg_reports})
    console(f"[thrusters] wrote: {ctx.module_dir / 'results.json'}", quiet=args.quiet)
    console(f"[thrusters] checks: {payload['counts']}", quiet=args.quiet)
    return 1 if payload["counts"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

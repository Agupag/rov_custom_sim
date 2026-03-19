#!/usr/bin/env python3
"""Cross-check docs/comments claims against detectable runtime reality."""

from __future__ import annotations

from pathlib import Path

from debug.common import CheckResult, console, emit_module_results, init_context, parse_common_args


def main() -> int:
    args = parse_common_args("Check runtime/docs consistency risks")
    ctx = init_context("consistency", args)

    import rov_sim

    results: list[CheckResult] = []

    readme_path = Path(rov_sim.DATA_DIR) / "README.md"
    project_context_path = Path(rov_sim.DATA_DIR) / "PROJECT_CONTEXT.md"
    legacy_asset = Path(rov_sim.DATA_DIR) / "Assembly_1.obj"
    legacy_backup = Path(rov_sim.DATA_DIR) / "rov_sim_backup.py"

    readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    # Claim check: if README mentions Assembly 1 but also documents multi-config behavior,
    # do not flag it as understated.
    mentions_assembly_claim = "Assembly 1.obj" in readme_text and "Assembly 1.gltf" in readme_text
    mentions_multi_config = any(
        token in readme_text
        for token in (
            "v1.obj",
            "v2.obj",
            "v3.obj",
            "THRUSTER_CONFIGS",
            "auto-discovery",
            "auto-discovers",
            "versioned CAD configs",
        )
    )
    multi_cfg_runtime = len(rov_sim.THRUSTER_CONFIGS) > 1
    warnings = []
    if mentions_assembly_claim and multi_cfg_runtime and not mentions_multi_config:
        warnings.append("README primary asset wording may understate multi-config auto-discovery behavior")

    results.append(
        CheckResult(
            name="readme_asset_claim_vs_runtime",
            status="warn" if warnings else "pass",
            summary="Compared README asset wording against runtime THRUSTER_CONFIGS discovery.",
            evidence={
                "readme_exists": readme_path.exists(),
                "mentions_assembly_assets": mentions_assembly_claim,
                "mentions_multi_config": mentions_multi_config,
                "runtime_config_count": len(rov_sim.THRUSTER_CONFIGS),
                "active_config": rov_sim.ACTIVE_CONFIG_NAME,
            },
            warnings=warnings,
        )
    )

    stale_items = []
    if legacy_asset.exists():
        stale_items.append("Assembly_1.obj")
    if legacy_backup.exists():
        stale_items.append("rov_sim_backup.py")

    documented_legacy = (
        "legacy" in readme_text.lower() and "rov_sim_backup.py" in readme_text
    )

    results.append(
        CheckResult(
            name="stale_or_legacy_items_present",
            status="warn" if stale_items and not documented_legacy else "pass",
            summary="Checked for known stale or potentially misleading legacy files.",
            evidence={
                "stale_items": stale_items,
                "documented_legacy_in_readme": documented_legacy,
            },
            warnings=["Legacy files can mislead debugging if treated as active runtime"] if stale_items and not documented_legacy else [],
        )
    )

    context_exists = project_context_path.exists()
    results.append(
        CheckResult(
            name="project_context_present",
            status="pass" if context_exists else "warn",
            summary="Verified presence of maintainer context document used for AI and human handoff.",
            evidence={"project_context_exists": context_exists},
        )
    )

    payload = emit_module_results(ctx, "consistency", results)
    console(f"[consistency] wrote: {ctx.module_dir / 'results.json'}", quiet=args.quiet)
    console(f"[consistency] checks: {payload['counts']}", quiet=args.quiet)
    return 1 if payload["counts"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

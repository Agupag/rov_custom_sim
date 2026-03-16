#!/usr/bin/env python3
"""Common helpers for debug and verification harness scripts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_ROOT = ROOT / "debug_artifacts"


@dataclass
class CheckResult:
    """Standardized result schema for all debug checks."""

    name: str
    status: str
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunContext:
    """Execution context shared across debug modules."""

    run_id: str
    artifacts_dir: Path
    module_dir: Path
    quiet: bool = False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str = "debug") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def parse_common_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-id", default=None, help="Optional run id override")
    parser.add_argument(
        "--artifacts-root",
        default=str(ARTIFACTS_ROOT),
        help="Root output directory for debug artifacts",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output",
    )
    return parser.parse_args()


def init_context(module_name: str, args: argparse.Namespace) -> RunContext:
    run_id = args.run_id or make_run_id("debug")
    artifacts_root = Path(args.artifacts_root)
    base_dir = ensure_dir(artifacts_root / run_id)
    module_dir = ensure_dir(base_dir / module_name)
    return RunContext(run_id=run_id, artifacts_dir=base_dir, module_dir=module_dir, quiet=args.quiet)


def emit_module_results(ctx: RunContext, module_name: str, results: list[CheckResult], extras: dict[str, Any] | None = None) -> dict[str, Any]:
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    warn = sum(1 for r in results if r.status == "warn")
    payload = {
        "module": module_name,
        "run_id": ctx.run_id,
        "generated_at_utc": utc_now_iso(),
        "counts": {
            "pass": passed,
            "fail": failed,
            "warn": warn,
            "total": len(results),
        },
        "results": [r.to_dict() for r in results],
        "extras": extras or {},
    }
    write_json(ctx.module_dir / "results.json", payload)
    write_text(ctx.module_dir / "results.md", render_markdown_summary(payload))
    return payload


def render_markdown_summary(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# {payload['module']} results")
    lines.append("")
    c = payload["counts"]
    lines.append(f"- Run: {payload['run_id']}")
    lines.append(f"- Generated (UTC): {payload['generated_at_utc']}")
    lines.append(f"- Pass: {c['pass']}")
    lines.append(f"- Warn: {c['warn']}")
    lines.append(f"- Fail: {c['fail']}")
    lines.append(f"- Total: {c['total']}")
    lines.append("")

    for item in payload["results"]:
        lines.append(f"## {item['name']} [{item['status'].upper()}]")
        lines.append(item.get("summary", ""))
        if item.get("warnings"):
            lines.append("")
            lines.append("Warnings:")
            for w in item["warnings"]:
                lines.append(f"- {w}")
        if item.get("errors"):
            lines.append("")
            lines.append("Errors:")
            for e in item["errors"]:
                lines.append(f"- {e}")
        lines.append("")
    return "\n".join(lines)


def console(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg)


def capture_environment_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd": str(Path.cwd()),
        "workspace_root": str(ROOT),
    }

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        info["git_commit_short"] = proc.stdout.strip() if proc.returncode == 0 else None
    except OSError:
        info["git_commit_short"] = None

    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        info["git_dirty"] = bool(proc.stdout.strip()) if proc.returncode == 0 else None
    except OSError:
        info["git_dirty"] = None

    return info


def load_module(name: str):
    # Import lazily so scripts can gather env data before pulling heavy deps.
    return __import__(name)

#!/usr/bin/env python3
"""Lightweight runtime event logger for simulator and panel instrumentation."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RuntimeEventLogger:
    """Append-only JSONL logger enabled by environment variable."""

    def __init__(self, source: str, enabled: bool, log_path: str | None = None):
        self.source = source
        self.enabled = enabled
        self.log_path = Path(log_path) if log_path else self._default_path()
        self._lock = threading.Lock()

    @staticmethod
    def from_environment(source: str) -> "RuntimeEventLogger":
        enabled = os.environ.get("ROV_DEBUG_EVENTS", "0") == "1"
        path = os.environ.get("ROV_DEBUG_EVENTS_FILE")
        return RuntimeEventLogger(source=source, enabled=enabled, log_path=path)

    @staticmethod
    def _default_path() -> Path:
        root = Path(__file__).resolve().parent.parent
        out_dir = root / "debug_artifacts" / "runtime"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return out_dir / f"runtime_events_{stamp}.jsonl"

    def emit(self, category: str, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        payload = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "source": self.source,
            "category": category,
            "event": event,
            "fields": fields,
        }
        line = json.dumps(payload, sort_keys=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

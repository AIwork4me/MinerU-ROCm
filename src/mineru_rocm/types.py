# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Run summary + per-page status types for mineru_rocm.

Local verbatim port of the engine types contract (RunSummary /
PageStatus), so the dispatcher no longer imports the platform engine. Same
field order and the same 8-key _run_stats.json shape the engine and the
existing tests expect. Stdlib-only.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass
class PageStatus:
    image: str
    status: str  # "ok" | "failed: <reason>" | "fallback: <reason>"
    error: str = ""
    seconds: float = 0.0
    attempts: int = 0


@dataclass
class RunSummary:
    count: int
    ok: int
    fail: int
    fallback: int
    limit_pages: int | None
    stats: list[PageStatus]
    engine: str = ""

    def to_run_stats(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "count": self.count,
            "ok": self.ok,
            "fail": self.fail,
            "fallback": self.fallback,
            "limit_pages": self.limit_pages,
            "engine": self.engine,
            "stats": [asdict(s) for s in self.stats],
        }

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_run_stats(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_run_stats(cls, path: Path) -> "RunSummary":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            count=d["count"], ok=d["ok"], fail=d["fail"], fallback=d["fallback"],
            limit_pages=d.get("limit_pages"),
            stats=[PageStatus(**s) for s in d.get("stats", [])],
            engine=d.get("engine", ""),
        )

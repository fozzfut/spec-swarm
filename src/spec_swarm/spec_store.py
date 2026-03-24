"""Persist specs to swarm-kb and local session files.

Storage layout:
    ~/.swarm-kb/spec/sessions/<session_id>/
        meta.json   -- session metadata
        specs.jsonl  -- one line per HardwareSpec
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .models import HardwareSpec, SpecSession, now_iso


def _default_base_path() -> Path:
    """Default storage path: ~/.swarm-kb/spec/sessions/"""
    return Path.home() / ".swarm-kb" / "spec" / "sessions"


class SpecStore:
    """Persist specs to swarm-kb and local session files."""

    def __init__(self, base_path: Path | None = None):
        self._base = base_path or _default_base_path()
        self._base.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # In-memory cache: session_id -> SpecSession
        self._sessions: dict[str, SpecSession] = {}
        self._load_existing_sessions()

    def _load_existing_sessions(self) -> None:
        """Load session metadata from disk on startup."""
        if not self._base.exists():
            return
        for sess_dir in self._base.iterdir():
            if not sess_dir.is_dir():
                continue
            meta_file = sess_dir / "meta.json"
            if not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                session = SpecSession(
                    id=meta.get("id", sess_dir.name),
                    project_path=meta.get("project_path", ""),
                    created_at=meta.get("created_at", ""),
                )
                # Load specs from jsonl
                specs_file = sess_dir / "specs.jsonl"
                if specs_file.exists():
                    for line in specs_file.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line:
                            try:
                                session.specs.append(HardwareSpec.from_dict(json.loads(line)))
                            except (json.JSONDecodeError, KeyError):
                                continue
                # Load findings
                findings_file = sess_dir / "findings.json"
                if findings_file.exists():
                    try:
                        session.findings = json.loads(findings_file.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, KeyError):
                        session.findings = []
                self._sessions[session.id] = session
            except (json.JSONDecodeError, OSError):
                continue

    def _session_dir(self, session_id: str) -> Path:
        """Get or create session directory."""
        # Prevent path traversal
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            raise ValueError(f"Invalid session_id: {session_id}")
        d = self._base / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def create_session(self, project_path: str) -> SpecSession:
        """Create a new spec analysis session."""
        with self._lock:
            session = SpecSession(project_path=project_path)
            self._sessions[session.id] = session
            self._save_meta(session)
            return session

    def get_session(self, session_id: str) -> SpecSession:
        """Get a session by ID. Raises KeyError if not found."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session {session_id} not found")
            import copy
            return copy.deepcopy(session)

    def list_sessions(self) -> list[dict]:
        """List all sessions with summary info."""
        with self._lock:
            result = []
            for sid, session in sorted(self._sessions.items()):
                result.append({
                    "session_id": session.id,
                    "project_path": session.project_path,
                    "created_at": session.created_at,
                    "spec_count": len(session.specs),
                    "finding_count": len(session.findings),
                })
            return result

    def add_spec(self, session_id: str, spec: HardwareSpec) -> None:
        """Add a hardware spec to a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session {session_id} not found")
            session.specs.append(spec)
            self._save_specs(session)

    def get_specs(self, session_id: str) -> list[HardwareSpec]:
        """Get all specs for a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session {session_id} not found")
            import copy
            return copy.deepcopy(session.specs)

    def add_finding(self, session_id: str, finding: dict) -> None:
        """Add a finding (conflict, missing info, etc.) to a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session {session_id} not found")
            finding.setdefault("timestamp", now_iso())
            session.findings.append(finding)
            self._save_findings(session)

    def get_findings(self, session_id: str) -> list[dict]:
        """Get all findings for a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session {session_id} not found")
            import copy
            return copy.deepcopy(session.findings)

    def _save_meta(self, session: SpecSession) -> None:
        """Save session metadata to disk."""
        d = self._session_dir(session.id)
        meta = {
            "id": session.id,
            "project_path": session.project_path,
            "created_at": session.created_at,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _save_specs(self, session: SpecSession) -> None:
        """Save all specs to jsonl file."""
        # Caller must hold self._lock
        d = self._session_dir(session.id)
        lines = [json.dumps(s.to_dict(), separators=(",", ":")) for s in session.specs]
        (d / "specs.jsonl").write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    def _save_findings(self, session: SpecSession) -> None:
        """Save findings to JSON file."""
        # Caller must hold self._lock
        d = self._session_dir(session.id)
        (d / "findings.json").write_text(
            json.dumps(session.findings, indent=2), encoding="utf-8"
        )

    def post_to_swarm_kb(self, session_id: str, tool: str, category: str, data: dict) -> bool:
        """Post data to swarm-kb if available. Returns True on success."""
        try:
            from swarm_kb.finding_writer import FindingWriter
            from swarm_kb.config import SuiteConfig

            config = SuiteConfig.load()
            writer = FindingWriter(tool, session_id, config)

            constraints = data.get("constraints", [])
            if constraints:
                writer.post_batch([
                    {
                        "type": "hw_constraint",
                        "category": c.get("category", category),
                        "component": c.get("component", ""),
                        "source": c.get("source", ""),
                        "constraint": c.get("constraint", ""),
                        "critical": c.get("critical", False),
                    }
                    for c in constraints
                ])
            else:
                writer.post({
                    "type": "hw_constraint",
                    "category": category,
                    "data": data,
                })
            return True
        except ImportError:
            return False
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to post to swarm-kb: %s", exc,
            )
            return False

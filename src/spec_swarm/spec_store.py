"""Persist specs to swarm-kb and local session files.

Storage layout:
    ~/.swarm-kb/spec/sessions/<session_id>/
        meta.json   -- session metadata
        specs.jsonl  -- one line per HardwareSpec
"""

from __future__ import annotations

import json
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
        d = self._base / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def create_session(self, project_path: str) -> SpecSession:
        """Create a new spec analysis session."""
        session = SpecSession(project_path=project_path)
        self._sessions[session.id] = session
        self._save_meta(session)
        return session

    def get_session(self, session_id: str) -> SpecSession:
        """Get a session by ID. Raises KeyError if not found."""
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        return self._sessions[session_id]

    def list_sessions(self) -> list[dict]:
        """List all sessions with summary info."""
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
        session = self.get_session(session_id)
        session.specs.append(spec)
        self._save_specs(session)

    def get_specs(self, session_id: str) -> list[HardwareSpec]:
        """Get all specs for a session."""
        session = self.get_session(session_id)
        return list(session.specs)

    def add_finding(self, session_id: str, finding: dict) -> None:
        """Add a finding (conflict, missing info, etc.) to a session."""
        session = self.get_session(session_id)
        finding.setdefault("timestamp", now_iso())
        session.findings.append(finding)
        self._save_findings(session)

    def get_findings(self, session_id: str) -> list[dict]:
        """Get all findings for a session."""
        session = self.get_session(session_id)
        return list(session.findings)

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
        d = self._session_dir(session.id)
        lines = [json.dumps(s.to_dict(), separators=(",", ":")) for s in session.specs]
        (d / "specs.jsonl").write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    def _save_findings(self, session: SpecSession) -> None:
        """Save findings to JSON file."""
        d = self._session_dir(session.id)
        (d / "findings.json").write_text(
            json.dumps(session.findings, indent=2), encoding="utf-8"
        )

    def post_to_swarm_kb(self, session_id: str, tool: str, category: str, data: dict) -> bool:
        """Post data to swarm-kb if available. Returns True on success."""
        try:
            from swarm_kb import KnowledgeBase
            kb = KnowledgeBase()
            kb.store(
                tool=tool,
                category=category,
                data=data,
                metadata={"session_id": session_id, "timestamp": now_iso()},
            )
            return True
        except ImportError:
            return False
        except Exception:
            return False

"""Multi-agent session management for spec verification."""

import json
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    CORRECTED = "corrected"


@dataclass
class SpecVerification:
    """An expert's verification of an extracted spec item."""

    id: str = ""
    spec_id: str = ""           # which HardwareSpec
    field_path: str = ""        # e.g. "registers[3].address", "protocols[0].speed"
    expert_role: str = ""
    status: str = "confirm"     # confirm | dispute | correct
    original_value: str = ""
    corrected_value: str = ""   # only if status=correct
    evidence: str = ""          # "Datasheet page 47, Table 12 shows 0x40021000"
    confidence: float = 0.9
    timestamp: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = "sv-" + secrets.token_hex(3)
        if not self.timestamp:
            self.timestamp = _now_iso()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "spec_id": self.spec_id,
            "field_path": self.field_path,
            "expert_role": self.expert_role,
            "status": self.status,
            "original_value": self.original_value,
            "corrected_value": self.corrected_value,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpecVerification":
        return cls(
            id=d.get("id", ""),
            spec_id=d.get("spec_id", ""),
            field_path=d.get("field_path", ""),
            expert_role=d.get("expert_role", ""),
            status=d.get("status", "confirm"),
            original_value=d.get("original_value", ""),
            corrected_value=d.get("corrected_value", ""),
            evidence=d.get("evidence", ""),
            confidence=d.get("confidence", 0.9),
            timestamp=d.get("timestamp", ""),
        )


@dataclass
class SpecClaim:
    """An expert claiming a spec/component for verification."""

    spec_id: str = ""
    expert_role: str = ""
    status: str = "active"  # active | released
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _now_iso()

    def to_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "expert_role": self.expert_role,
            "status": self.status,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpecClaim":
        return cls(
            spec_id=d.get("spec_id", ""),
            expert_role=d.get("expert_role", ""),
            status=d.get("status", "active"),
            timestamp=d.get("timestamp", ""),
        )


def _default_sessions_dir() -> Path:
    return Path.home() / ".swarm-kb" / "spec" / "verification_sessions"


class SpecSessionManager:
    """Manages multi-agent spec verification sessions."""

    def __init__(self, sessions_dir: Optional[Path] = None):
        self._dir = sessions_dir or _default_sessions_dir()
        self._sessions: dict[str, dict] = {}  # session metadata
        self._verifications: dict[str, list[SpecVerification]] = {}
        self._claims: dict[str, list[SpecClaim]] = {}
        self._messages: dict[str, list[dict]] = {}
        self._phase_done: dict[str, dict[str, set[int]]] = {}
        self._lock = threading.Lock()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._load_all()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, project_path: str = "", name: str = "") -> str:
        """Start a new spec verification session. Returns session_id."""
        with self._lock:
            session_id = self._generate_session_id()
            now = _now_iso()
            self._sessions[session_id] = {
                "session_id": session_id,
                "name": name or session_id,
                "project_path": project_path,
                "status": "active",
                "created": now,
                "ended": "",
            }
            self._verifications[session_id] = []
            self._claims[session_id] = []
            self._messages[session_id] = []
            self._phase_done[session_id] = {}
            self._save_session(session_id)
            return session_id

    def end_session(self, session_id: str) -> dict:
        """End session, generate verification summary."""
        with self._lock:
            meta = self._require_session(session_id)
            meta["status"] = "ended"
            meta["ended"] = _now_iso()
            self._save_session(session_id)
            return self._build_summary_unlocked(session_id)

    def get_session(self, session_id: str) -> dict:
        """Get session metadata + stats."""
        with self._lock:
            meta = dict(self._require_session(session_id))
            verifications = self._verifications.get(session_id, [])
            meta["stats"] = {
                "total_verifications": len(verifications),
                "confirmed": sum(1 for v in verifications if v.status == "confirm"),
                "disputed": sum(1 for v in verifications if v.status == "dispute"),
                "corrected": sum(1 for v in verifications if v.status == "correct"),
                "active_claims": sum(
                    1 for c in self._claims.get(session_id, [])
                    if c.status == "active"
                ),
                "total_messages": len(self._messages.get(session_id, [])),
            }
            return meta

    def list_sessions(self) -> list[dict]:
        """List all verification sessions."""
        with self._lock:
            return [dict(m) for m in self._sessions.values()]

    # ------------------------------------------------------------------
    # Claims
    # ------------------------------------------------------------------

    def claim_spec(self, session_id: str, spec_id: str, expert_role: str) -> dict:
        """Claim a component/spec for verification. Prevents duplicate work."""
        with self._lock:
            self._require_session(session_id)
            claims = self._claims[session_id]

            # Check if already actively claimed by same expert
            for c in claims:
                if c.spec_id == spec_id and c.status == "active":
                    if c.expert_role == expert_role:
                        return {"ok": True, "already_claimed": True, "claim": c.to_dict()}
                    # Different expert -- allow (multiple experts can verify same spec)
                    # but inform about existing claims
                    pass

            claim = SpecClaim(spec_id=spec_id, expert_role=expert_role, status="active")
            claims.append(claim)
            self._save_session(session_id)
            return {"ok": True, "already_claimed": False, "claim": claim.to_dict()}

    def release_spec(self, session_id: str, spec_id: str, expert_role: str) -> dict:
        """Release a claimed component."""
        with self._lock:
            self._require_session(session_id)
            claims = self._claims[session_id]

            for c in claims:
                if (
                    c.spec_id == spec_id
                    and c.expert_role == expert_role
                    and c.status == "active"
                ):
                    c.status = "released"
                    self._save_session(session_id)
                    return {"ok": True, "claim": c.to_dict()}

            return {"ok": False, "error": f"No active claim found for {spec_id} by {expert_role}"}

    def get_claims(self, session_id: str) -> list[dict]:
        """Get all claims for a session."""
        with self._lock:
            self._require_session(session_id)
            return [c.to_dict() for c in self._claims[session_id]]

    # ------------------------------------------------------------------
    # Verifications
    # ------------------------------------------------------------------

    def post_verification(self, session_id: str, verification: SpecVerification) -> str:
        """Post a verification result (confirm, dispute, or correct). Returns verification id."""
        with self._lock:
            self._require_session(session_id)
            if not verification.id:
                verification.__post_init__()
            self._verifications[session_id].append(verification)
            self._save_session(session_id)
            return verification.id

    def get_verifications(
        self,
        session_id: str,
        spec_id: str = "",
        expert_role: str = "",
        status: str = "",
    ) -> list[dict]:
        """Query verifications with filters."""
        with self._lock:
            self._require_session(session_id)
            results = self._verifications[session_id]
            if spec_id:
                results = [v for v in results if v.spec_id == spec_id]
            if expert_role:
                results = [v for v in results if v.expert_role == expert_role]
            if status:
                results = [v for v in results if v.status == status]
            return [v.to_dict() for v in results]

    def get_verification_status(self, session_id: str, spec_id: str) -> dict:
        """Get aggregated verification status for a spec.

        Returns dict with total_checks, confirmed, disputed, corrected counts,
        experts who verified, overall status, and lists of disputes/corrections.
        """
        with self._lock:
            self._require_session(session_id)
            verifications = [
                v for v in self._verifications[session_id]
                if v.spec_id == spec_id
            ]

            confirmed = sum(1 for v in verifications if v.status == "confirm")
            disputed = sum(1 for v in verifications if v.status == "dispute")
            corrected = sum(1 for v in verifications if v.status == "correct")
            total = len(verifications)

            experts = sorted({v.expert_role for v in verifications})

            # Determine overall status
            if total == 0:
                overall = VerificationStatus.UNVERIFIED.value
            elif disputed > 0:
                overall = VerificationStatus.DISPUTED.value
            elif corrected > 0:
                overall = VerificationStatus.CORRECTED.value
            else:
                overall = VerificationStatus.VERIFIED.value

            disputes = [
                {
                    "field": v.field_path,
                    "expert": v.expert_role,
                    "evidence": v.evidence,
                    "original_value": v.original_value,
                }
                for v in verifications if v.status == "dispute"
            ]

            corrections = [
                {
                    "field": v.field_path,
                    "old_value": v.original_value,
                    "new_value": v.corrected_value,
                    "expert": v.expert_role,
                    "evidence": v.evidence,
                }
                for v in verifications if v.status == "correct"
            ]

            return {
                "spec_id": spec_id,
                "total_checks": total,
                "confirmed": confirmed,
                "disputed": disputed,
                "corrected": corrected,
                "experts_verified": experts,
                "overall_status": overall,
                "disputes": disputes,
                "corrections": corrections,
            }

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def send_message(
        self,
        session_id: str,
        sender: str,
        recipient: str,
        content: str,
        context_id: str = "",
    ) -> str:
        """Send a message to another expert. Returns message id."""
        with self._lock:
            self._require_session(session_id)
            msg_id = "msg-" + secrets.token_hex(3)
            msg = {
                "id": msg_id,
                "sender": sender,
                "recipient": recipient,
                "content": content,
                "context_id": context_id,
                "timestamp": _now_iso(),
            }
            self._messages[session_id].append(msg)
            self._save_session(session_id)
            return msg_id

    def get_inbox(self, session_id: str, expert_role: str) -> list[dict]:
        """Get pending messages for a spec expert."""
        with self._lock:
            self._require_session(session_id)
            results: list[dict] = []
            for msg in self._messages[session_id]:
                if msg["recipient"] == expert_role or msg["recipient"] == "all":
                    results.append(msg)
            return results

    def broadcast(self, session_id: str, sender: str, content: str) -> str:
        """Broadcast a message to all spec experts. Returns message id."""
        return self.send_message(session_id, sender, "all", content)

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def mark_phase_done(self, session_id: str, expert_role: str, phase: int) -> dict:
        """Mark that an expert completed a verification phase."""
        with self._lock:
            self._require_session(session_id)
            phases = self._phase_done.setdefault(session_id, {})
            expert_phases = phases.setdefault(expert_role, set())
            expert_phases.add(phase)
            self._save_session(session_id)
            return {
                "ok": True,
                "expert_role": expert_role,
                "phase": phase,
                "phases_done": {k: sorted(v) for k, v in phases.items()},
            }

    def check_phase_ready(self, session_id: str, phase: int) -> dict:
        """Check if all experts completed a verification phase."""
        with self._lock:
            self._require_session(session_id)
            phases = self._phase_done.get(session_id, {})
            claims = self._claims.get(session_id, [])

            # Participating experts = those with active claims
            active_experts = {c.expert_role for c in claims if c.status == "active"}
            if not active_experts:
                return {"ready": True, "phase": phase, "waiting_on": [], "active_experts": []}

            done_experts = {
                role for role, done_set in phases.items() if phase in done_set
            }
            waiting = sorted(active_experts - done_experts)
            return {
                "ready": len(waiting) == 0,
                "phase": phase,
                "waiting_on": waiting,
                "done": sorted(done_experts & active_experts),
                "active_experts": sorted(active_experts),
            }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self, session_id: str) -> dict:
        """Generate verification summary."""
        with self._lock:
            self._require_session(session_id)
            return self._build_summary_unlocked(session_id)

    def _build_summary_unlocked(self, session_id: str) -> dict:
        """Internal summary builder (caller must hold lock)."""
        meta = self._sessions[session_id]
        verifications = self._verifications.get(session_id, [])
        claims = self._claims.get(session_id, [])
        messages = self._messages.get(session_id, [])

        confirmed = sum(1 for v in verifications if v.status == "confirm")
        disputed = sum(1 for v in verifications if v.status == "dispute")
        corrected = sum(1 for v in verifications if v.status == "correct")
        total = len(verifications)

        # Aggregate by spec_id
        spec_ids = sorted({v.spec_id for v in verifications})
        per_spec: list[dict] = []
        for sid in spec_ids:
            sv = [v for v in verifications if v.spec_id == sid]
            c = sum(1 for v in sv if v.status == "confirm")
            d = sum(1 for v in sv if v.status == "dispute")
            co = sum(1 for v in sv if v.status == "correct")
            if d > 0:
                status = VerificationStatus.DISPUTED.value
            elif co > 0:
                status = VerificationStatus.CORRECTED.value
            elif c > 0:
                status = VerificationStatus.VERIFIED.value
            else:
                status = VerificationStatus.UNVERIFIED.value
            per_spec.append({
                "spec_id": sid,
                "total": len(sv),
                "confirmed": c,
                "disputed": d,
                "corrected": co,
                "status": status,
            })

        experts_involved = sorted(
            {v.expert_role for v in verifications}
            | {c.expert_role for c in claims}
            | {m["sender"] for m in messages if m.get("sender")}
        )

        corrections = [
            {
                "field": v.field_path,
                "spec_id": v.spec_id,
                "old_value": v.original_value,
                "new_value": v.corrected_value,
                "expert": v.expert_role,
                "evidence": v.evidence,
            }
            for v in verifications if v.status == "correct"
        ]

        disputes = [
            {
                "field": v.field_path,
                "spec_id": v.spec_id,
                "expert": v.expert_role,
                "evidence": v.evidence,
                "original_value": v.original_value,
            }
            for v in verifications if v.status == "dispute"
        ]

        return {
            "session_id": session_id,
            "name": meta.get("name", ""),
            "status": meta.get("status", ""),
            "created": meta.get("created", ""),
            "ended": meta.get("ended", ""),
            "total_verifications": total,
            "confirmed": confirmed,
            "disputed": disputed,
            "corrected": corrected,
            "confirmation_rate": round(confirmed / total * 100, 1) if total > 0 else 0.0,
            "per_spec": per_spec,
            "corrections": corrections,
            "disputes": disputes,
            "experts_involved": experts_involved,
            "total_messages": len(messages),
            "active_claims": sum(1 for c in claims if c.status == "active"),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_session(self, session_id: str) -> None:
        """Persist session state to disk. Caller must hold self._lock."""
        sdir = self._dir / session_id
        sdir.mkdir(parents=True, exist_ok=True)

        # meta.json
        meta = dict(self._sessions[session_id])
        # Serialize phase_done (sets -> lists)
        phases = self._phase_done.get(session_id, {})
        meta["phases_done"] = {k: sorted(v) for k, v in phases.items()}
        (sdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # verifications.jsonl
        with open(sdir / "verifications.jsonl", "w", encoding="utf-8") as f:
            for v in self._verifications.get(session_id, []):
                f.write(json.dumps(v.to_dict()) + "\n")

        # claims.jsonl
        with open(sdir / "claims.jsonl", "w", encoding="utf-8") as f:
            for c in self._claims.get(session_id, []):
                f.write(json.dumps(c.to_dict()) + "\n")

        # messages.jsonl
        with open(sdir / "messages.jsonl", "w", encoding="utf-8") as f:
            for m in self._messages.get(session_id, []):
                f.write(json.dumps(m) + "\n")

    def _load_all(self) -> None:
        """Scan sessions directory and load existing sessions from disk."""
        if not self._dir.exists():
            return
        for d in sorted(self._dir.iterdir()):
            if d.is_dir() and (d / "meta.json").exists():
                sid = d.name
                if sid not in self._sessions:
                    self._load_session(sid)

    def _load_session(self, session_id: str) -> None:
        """Load a single session from disk. Caller must hold self._lock."""
        sdir = self._dir / session_id
        if not sdir.exists():
            return

        # meta.json
        meta_path = sdir / "meta.json"
        if not meta_path.exists():
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Restore phases_done as sets
        phases_raw = meta.pop("phases_done", {})
        self._sessions[session_id] = meta
        self._phase_done[session_id] = {k: set(v) for k, v in phases_raw.items()}

        # verifications.jsonl
        self._verifications[session_id] = []
        vpath = sdir / "verifications.jsonl"
        if vpath.exists():
            for line in vpath.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        self._verifications[session_id].append(
                            SpecVerification.from_dict(json.loads(line))
                        )
                    except (json.JSONDecodeError, KeyError):
                        continue

        # claims.jsonl
        self._claims[session_id] = []
        cpath = sdir / "claims.jsonl"
        if cpath.exists():
            for line in cpath.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        self._claims[session_id].append(
                            SpecClaim.from_dict(json.loads(line))
                        )
                    except (json.JSONDecodeError, KeyError):
                        continue

        # messages.jsonl
        self._messages[session_id] = []
        mpath = sdir / "messages.jsonl"
        if mpath.exists():
            for line in mpath.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        self._messages[session_id].append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_session(self, session_id: str) -> dict:
        """Return session metadata or raise KeyError."""
        if session_id not in self._sessions:
            raise KeyError(f"Verification session {session_id!r} not found")
        return self._sessions[session_id]

    def _generate_session_id(self) -> str:
        """Generate a session ID like vsess-2026-03-24-001."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prefix = f"vsess-{today}-"

        existing = [
            sid for sid in self._sessions if sid.startswith(prefix)
        ]
        if self._dir.exists():
            for d in self._dir.iterdir():
                if d.is_dir() and d.name.startswith(prefix) and d.name not in existing:
                    existing.append(d.name)

        max_n = 0
        for sid in existing:
            suffix = sid[len(prefix):]
            try:
                n = int(suffix)
                if n > max_n:
                    max_n = n
            except ValueError:
                pass
        return f"{prefix}{max_n + 1:03d}"

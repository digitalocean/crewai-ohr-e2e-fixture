"""Duck-typed conversational Flow for harness_framework L1 E2E.

No LLM calls. Exercises:
  - multi-turn memory keyed by session_id
  - HumanFeedbackPending → from_pending → resume_async loop
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Ctx:
    flow_id: str


@dataclass
class HumanFeedbackPending:
    """Mirrors crewai.flow.async_feedback.types.HumanFeedbackPending shape."""

    context: _Ctx
    message: str = "Human feedback required"
    callback_info: dict[str, Any] = field(default_factory=dict)


# Process-local session memory (survives within one mediator process / guest).
_SESSIONS: dict[str, list[str]] = {}
_PENDING: dict[str, dict[str, Any]] = {}


class E2EFlow:
    conversational = True

    def __init__(self) -> None:
        self._session_id: str | None = None

    def handle_turn(self, message: str, session_id: str | None = None) -> Any:
        sid = session_id or "default"
        self._session_id = sid
        history = _SESSIONS.setdefault(sid, [])
        history.append(f"user:{message}")

        lower = message.lower()
        if "confirm" in lower or "human feedback" in lower or "hitl" in lower:
            flow_id = f"flow-{sid}-hitl-{len(history)}"
            _PENDING[flow_id] = {"session_id": sid, "prompt": message}
            return HumanFeedbackPending(
                context=_Ctx(flow_id=flow_id),
                message="Please confirm the codeword.",
            )

        if "codeword" in lower and "what" in lower:
            # Multi-turn recall: look for prior "remember the codeword X"
            codeword = "unknown"
            for entry in history:
                if entry.startswith("user:") and "remember" in entry.lower() and "codeword" in entry.lower():
                    # "remember the codeword ALPHA"
                    parts = entry.split()
                    codeword = parts[-1].rstrip(".")
            reply = f"The codeword was {codeword}."
            history.append(f"assistant:{reply}")
            return reply

        reply = f"Noted: {message}"
        history.append(f"assistant:{reply}")
        return reply

    @classmethod
    def from_pending(cls, flow_id: str) -> E2EFlow:
        meta = _PENDING.get(flow_id)
        if not meta:
            raise ValueError(f"unknown pending flow_id={flow_id}")
        flow = cls()
        flow._session_id = meta["session_id"]
        flow._resume_flow_id = flow_id  # type: ignore[attr-defined]
        return flow

    async def resume_async(self, feedback: str = "") -> Any:
        flow_id = getattr(self, "_resume_flow_id", None)
        if not flow_id or flow_id not in _PENDING:
            raise ValueError("no pending HITL context")
        meta = _PENDING.pop(flow_id)
        sid = meta["session_id"]
        history = _SESSIONS.setdefault(sid, [])
        reply = f"HITL confirmed with feedback: {feedback or '(empty)'}"
        history.append(f"assistant:{reply}")
        return reply

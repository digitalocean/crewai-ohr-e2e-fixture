"""Real CrewAI conversational Flow for OHR L1 E2E.

Uses the real `crewai.Flow` base, real `HumanFeedbackPending`, and emits real
`crewai_event_bus` events so the mediator's bus listener produces token/tool
frames. Logic is deterministic (no LLM calls) so the E2E stays fast and
offline-capable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from crewai.events import (
    LLMStreamChunkEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
    crewai_event_bus,
)
from crewai.flow.async_feedback.types import HumanFeedbackPending, PendingFeedbackContext
from crewai.flow.flow import Flow, start

# Process-local session memory (survives within one mediator process / guest).
_SESSIONS: dict[str, list[str]] = {}
_PENDING: dict[str, dict[str, Any]] = {}


class E2EFlow(Flow):
    """Conversational Flow subclass — proves L1 discovery against real crewai."""

    conversational = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._session_id: str | None = None
        self._resume_flow_id: str | None = None

    @start()
    def begin(self) -> str:
        # Required Flow entrypoint; conversational turns go through handle_turn.
        return "ready"

    def handle_turn(
        self,
        message: str,
        *,
        session_id: str | None = None,
        **_kwargs: Any,
    ) -> Any:
        """Deterministic turn handler matching the OHR E2E script prompts."""
        sid = session_id or "default"
        self._session_id = sid
        history = _SESSIONS.setdefault(sid, [])
        history.append(f"user:{message}")

        # Prove the mediator bus path: emit real crewai events the listener maps
        # to token / tool.started / tool.completed frames.
        self._emit_bus_activity(message)

        lower = message.lower()
        if "confirm" in lower or "human feedback" in lower or "hitl" in lower:
            flow_id = f"flow-{sid}-hitl-{len(history)}"
            _PENDING[flow_id] = {"session_id": sid, "prompt": message}
            return HumanFeedbackPending(
                context=PendingFeedbackContext(
                    flow_id=flow_id,
                    flow_class=f"{type(self).__module__}.{type(self).__name__}",
                    method_name="handle_turn",
                    method_output=None,
                    message="Please confirm the codeword.",
                ),
                message="Please confirm the codeword.",
            )

        if "codeword" in lower and "what" in lower:
            codeword = "unknown"
            for entry in history:
                if (
                    entry.startswith("user:")
                    and "remember" in entry.lower()
                    and "codeword" in entry.lower()
                ):
                    parts = entry.split()
                    codeword = parts[-1].rstrip(".")
            reply = f"The codeword was {codeword}."
            history.append(f"assistant:{reply}")
            return reply

        reply = f"Noted: {message}"
        history.append(f"assistant:{reply}")
        return reply

    @classmethod
    def from_pending(cls, flow_id: str, persistence: Any = None, **kwargs: Any) -> E2EFlow:
        """Restore a HITL-paused turn without CrewAI SQLite persistence.

        The mediator parks on returned HumanFeedbackPending and resumes via
        this classmethod. We keep a process-local map so the E2E stays
        self-contained (no LLM, no durable store required for the pause).
        """
        del persistence  # unused — E2E HITL is in-process
        meta = _PENDING.get(flow_id)
        if not meta:
            raise ValueError(f"unknown pending flow_id={flow_id}")
        flow = cls(**kwargs)
        flow._session_id = meta["session_id"]
        flow._resume_flow_id = flow_id
        return flow

    async def resume_async(self, feedback: str = "") -> Any:
        flow_id = self._resume_flow_id
        if not flow_id or flow_id not in _PENDING:
            raise ValueError("no pending HITL context")
        meta = _PENDING.pop(flow_id)
        sid = meta["session_id"]
        history = _SESSIONS.setdefault(sid, [])
        reply = f"HITL confirmed with feedback: {feedback or '(empty)'}"
        history.append(f"assistant:{reply}")
        self._resume_flow_id = None
        return reply

    def _emit_bus_activity(self, message: str) -> None:
        """Fire real crewai events so harness_framework.bus produces frames."""
        now = datetime.now(timezone.utc)
        call_id = f"e2e-{id(self)}"
        crewai_event_bus.emit(
            self, LLMStreamChunkEvent(call_id=call_id, chunk=f"thinking about: {message[:40]}")
        )
        crewai_event_bus.emit(
            self,
            ToolUsageStartedEvent(tool_name="e2e_echo", tool_args={"text": message[:80]}),
        )
        crewai_event_bus.emit(
            self,
            ToolUsageFinishedEvent(
                tool_name="e2e_echo",
                tool_args={"text": message[:80]},
                started_at=now,
                finished_at=now,
                output="ok",
            ),
        )

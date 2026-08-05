# crewai-ohr-e2e-fixture

Real **CrewAI** conversational Flow for DigitalOcean OHR `crewai` L1 E2E.

- Subclasses `crewai.Flow` with `conversational = True`
- Returns real `HumanFeedbackPending` for HITL
- Emits real `crewai_event_bus` events (token / tool frames)

Deterministic (no LLM calls) so the multi-turn + HITL script stays offline-capable.

Entrypoint: `e2e_flow.flow:E2EFlow`

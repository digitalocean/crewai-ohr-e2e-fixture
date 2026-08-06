# crewai-ohr-e2e-fixture

Real **CrewAI Crew** for DigitalOcean OHR `crewai` L1 kickoff E2E.

- Two agents in `Process.sequential`: Researcher → Writer
- Driven by `Crew.akickoff(inputs={"topic": …})` (mediator default path)
- Uses `OPENAI_API_KEY` + `MODEL` from the session env (not baked into the repo)

Entrypoint: `e2e_crew.crew:crew` (see `crewai.json`).

The older conversational Flow fixture remains under `e2e_flow/` for the
opt-in `FRAMEWORK_PROVIDER=crewai-flow` path.

"""Two-agent sequential CrewAI Crew for OHR kickoff E2E.

Uses a real LLM from the session env:
  OPENAI_API_KEY (required)
  MODEL / OPENAI_MODEL (default gpt-4o)
  OPENAI_BASE_URL / OPENAI_API_BASE (optional — e.g. DO inference)

Action Gateway tool wiring (HARNESS_MCP_URL_<NAME> scenario):
  The platform puts each attached MCP server's endpoint in its own env var
  (cthulhu#177963); for `do.actions` that is HARNESS_MCP_URL_DO_ACTIONS. This
  is the customer-shaped code: read the URL and hand it to CrewAI's own
  `crewai_tools.MCPServerAdapter`. No decoding, no DigitalOcean library.
"""

from __future__ import annotations

import os

from crewai import Agent, Crew, LLM, Process, Task

_model = os.environ.get("MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o"
_base = (
    os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("OPENAI_API_BASE")
    or ""
).strip()

_llm_kwargs: dict = {
    "model": _model,
    "api_key": os.environ.get("OPENAI_API_KEY"),
}
if _base:
    _llm_kwargs["base_url"] = _base

_llm = LLM(**_llm_kwargs)


def _load_action_gateway_tools() -> list:
    """Attach the session's Action Gateway tools, if the spec declared them.

    Best-effort: a missing env var (running outside Managed Agents) or an
    unreachable gateway yields no tools rather than failing the Crew.
    """
    url = os.environ.get("HARNESS_MCP_URL_DO_ACTIONS")
    if not url:
        print("[e2e_crew] HARNESS_MCP_URL_DO_ACTIONS unset; no Action Gateway tools")
        return []
    try:
        from crewai_tools import MCPServerAdapter

        adapter = MCPServerAdapter({"url": url, "transport": "streamable-http"})
        tools = list(adapter.tools)
    except Exception as exc:  # noqa: BLE001 - defensive, log and continue
        print(f"[e2e_crew] Action Gateway tool wiring failed: {exc}")
        return []
    print(f"[e2e_crew] Action Gateway tools: {[t.name for t in tools]}")
    return tools


_ag_tools = _load_action_gateway_tools()

researcher = Agent(
    role="Researcher",
    goal="Gather 2-3 crisp facts about the given topic",
    backstory="You are a concise researcher. Prefer short bullet facts.",
    llm=_llm,
    tools=_ag_tools,
    allow_delegation=False,
    verbose=True,
)

writer = Agent(
    role="Writer",
    goal="Turn research into one short paragraph",
    backstory="You write clear, short summaries for engineers.",
    llm=_llm,
    allow_delegation=False,
    verbose=True,
)

_research_instructions = (
    "Research this topic and return exactly 2-3 short bullet facts. "
    "Topic: {topic}"
)
if _ag_tools:
    _research_instructions += (
        " Use your web search tool at least once to find one current, "
        "verifiable fact before answering."
    )

research_task = Task(
    description=_research_instructions,
    expected_output="2-3 bullet points of key facts (no preamble)",
    agent=researcher,
)

write_task = Task(
    description=(
        "Using the research, write one short paragraph summarizing the topic. "
        "End the paragraph with the exact marker CREW_E2E_OK. Topic: {topic}"
    ),
    expected_output="One short paragraph ending with CREW_E2E_OK",
    agent=writer,
    context=[research_task],
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    stream=True,
    verbose=True,
)

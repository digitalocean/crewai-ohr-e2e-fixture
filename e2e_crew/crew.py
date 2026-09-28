"""Two-agent sequential CrewAI Crew for OHR kickoff E2E.

Uses a real LLM from the session env:
  OPENAI_API_KEY (required)
  MODEL / OPENAI_MODEL (default gpt-4o)
  OPENAI_BASE_URL / OPENAI_API_BASE (optional — e.g. DO inference)

Action Gateway tool wiring (added for the AG integration scenario):
  The platform mints one session-pinned MCP endpoint per declared
  `do.actions` server and ships it to the guest as `HARNESS_MCP_SERVERS`
  (base64 JSON array), a reserved env key customers can't set themselves.
  CrewAI has no built-in attach hook for this today, so — same as any
  customer wiring their own agent up to it — we decode the env var
  ourselves and hand the discovered tools to the Researcher via
  `crewai_tools.MCPServerAdapter`.
"""

from __future__ import annotations

import base64
import json
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
    """Discover the `do_actions` MCP server (if the manifest declared one
    under `spec.tools.mcpServers`) and wrap it as CrewAI tools.

    Best-effort: any failure here (env absent, package missing, gateway
    unreachable) just yields no tools rather than failing the whole Crew,
    matching how a customer would defensively wire an optional tool.
    """
    raw = os.environ.get("HARNESS_MCP_SERVERS", "")
    if not raw:
        return []
    try:
        servers = json.loads(base64.b64decode(raw))
    except Exception as exc:  # noqa: BLE001 - defensive, log and continue
        print(f"[e2e_crew] HARNESS_MCP_SERVERS decode failed: {exc}")
        return []

    tools: list = []
    for server in servers:
        if server.get("name") != "do_actions":
            continue
        try:
            from crewai_tools import MCPServerAdapter

            adapter = MCPServerAdapter(
                {"url": server["url"], "transport": "streamable-http"}
            )
            discovered = list(adapter.tools)
            print(
                f"[e2e_crew] do_actions MCP tools discovered: "
                f"{[t.name for t in discovered]}"
            )
            tools.extend(discovered)
        except Exception as exc:  # noqa: BLE001 - defensive, log and continue
            print(f"[e2e_crew] do_actions MCP tool wiring failed: {exc}")
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

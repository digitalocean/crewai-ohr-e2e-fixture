"""Two-agent sequential CrewAI Crew for OHR kickoff E2E.

Uses a real LLM from the session env:
  OPENAI_API_KEY (required)
  MODEL / OPENAI_MODEL (default gpt-4o)
  OPENAI_BASE_URL / OPENAI_API_BASE (optional — e.g. DO inference)

Action Gateway tool wiring (added for the AG integration scenario):
  The platform mints one session-pinned MCP endpoint per declared
  `do.actions` server and ships it to the guest as `HARNESS_MCP_SERVERS`
  (base64 JSON array), a reserved env key customers can't set themselves.
  This is the raw version of what `harness_crewai.tools.action_gateway_tools()`
  does (plano), inlined so it runs on the current prod image: decode the env
  var, list the `do_actions` server's tools with crewai's own MCP client (no
  crewai-tools), and give each tool the name the server advertises — crewai
  otherwise prefixes it with the session URL and truncates to 64 chars, which
  erases the tool name. The manifest must `preload_tools` a tool for the
  gateway to list it by name.
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
    """Return the `do_actions` MCP server's tools as CrewAI tools.

    Best-effort: any failure here (env absent, gateway unreachable) just
    yields no tools rather than failing the whole Crew, matching how a
    customer would defensively wire an optional tool.
    """
    raw = os.environ.get("HARNESS_MCP_SERVERS", "")
    if not raw:
        return []
    try:
        servers = json.loads(base64.b64decode(raw))
    except Exception as exc:  # noqa: BLE001 - defensive, log and continue
        print(f"[e2e_crew] HARNESS_MCP_SERVERS decode failed: {exc}")
        return []

    url = next((s["url"] for s in servers if s.get("name") == "do_actions"), None)
    if not url:
        return []
    try:
        from crewai.mcp.config import MCPServerHTTP
        from crewai.mcp.tool_resolver import MCPToolResolver
        from crewai.utilities.logger import Logger

        tools = MCPToolResolver(agent=None, logger=Logger()).resolve(
            [MCPServerHTTP(url=url)]
        )
    except Exception as exc:  # noqa: BLE001 - defensive, log and continue
        print(f"[e2e_crew] do_actions MCP tool listing failed: {exc}")
        return []
    for tool in tools:
        tool.name = tool.original_tool_name
    print(f"[e2e_crew] do_actions MCP tools: {[t.name for t in tools]}")
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

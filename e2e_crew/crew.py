"""Two-agent sequential CrewAI Crew for OHR kickoff E2E.

Uses a real LLM from the session env:
  OPENAI_API_KEY (required)
  MODEL / OPENAI_MODEL (default gpt-4o)
  OPENAI_BASE_URL / OPENAI_API_BASE (optional — e.g. DO inference)

Action Gateway tool wiring (native-MCP scenario):
  What a customer would write if the platform hands each MCP server's URL to
  the guest in its own env var (HARNESS_MCP_URL_<NAME>, proposed): pass it
  straight to crewai's built-in `Agent(mcps=[MCPServerHTTP(url=...)])`, no
  decoding, no helper library. Tools are listed per task at kickoff. The
  manifest must `preload_tools` a tool for the gateway to list it by name.
"""

from __future__ import annotations

import base64
import json
import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai.mcp import MCPServerHTTP

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


def _action_gateway_url() -> str | None:
    url = os.environ.get("HARNESS_MCP_URL_DO_ACTIONS")
    if url:
        return url
    # TEST-ONLY SHIM: prod doesn't emit HARNESS_MCP_URL_<NAME> yet, so derive
    # it from HARNESS_MCP_SERVERS. Customer code would stop at the line above.
    raw = os.environ.get("HARNESS_MCP_SERVERS", "")
    if not raw:
        return None
    servers = json.loads(base64.b64decode(raw))
    return next((s["url"] for s in servers if s.get("name") == "do_actions"), None)


_ag_url = _action_gateway_url()

researcher = Agent(
    role="Researcher",
    goal="Gather 2-3 crisp facts about the given topic",
    backstory="You are a concise researcher. Prefer short bullet facts.",
    llm=_llm,
    mcps=[MCPServerHTTP(url=_ag_url)] if _ag_url else [],
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
if _ag_url:
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

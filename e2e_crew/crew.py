"""Two-agent sequential CrewAI Crew for OHR kickoff E2E.

Uses a real LLM from the session env:
  OPENAI_API_KEY (required)
  MODEL / OPENAI_MODEL (default gpt-4o)
  OPENAI_BASE_URL / OPENAI_API_BASE (optional — e.g. DO inference)

Action Gateway tool wiring (Action Gateway SDK integration scenario):
  Supersedes the HARNESS_MCP_SERVERS + crewai_tools.MCPServerAdapter flow
  from MARSOHS-1691. pydo (the official DO Python client) now ships a
  purpose-built `pydo.action_gateway.ActionGatewayClient`: it mints a
  policy-scoped Action Gateway session directly against the DO API and
  exposes plain tool discovery (`session.tools.list`) and invocation
  (`session.tools.call`) — no MCP transport, no base64 env-var decoding,
  no `crewai_tools` dependency.

  This changes the trust model versus the platform-minted, credential-free
  `do_actions` MCP endpoint: the crew process itself now holds a real
  DIGITALOCEAN_TOKEN (delivered via spec.secrets, never spec.env — env
  values are debug-readable inside the sandbox) and mints its own session,
  so the tool allowlist is expressed here (ACTION_GATEWAY_TOOLS) instead
  of the manifest's spec.tools.mcpServers / spec.permissions.rules.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

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

_JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _args_model(tool_name: str, schema: Dict[str, Any]) -> Optional[type]:
    """Build a pydantic args model from an Action Gateway tool's inputSchema."""
    from pydantic import create_model

    properties = (schema or {}).get("properties") or {}
    if not properties:
        return None
    required = set((schema or {}).get("required") or [])
    fields: Dict[str, Any] = {}
    for prop, spec in properties.items():
        py_type = _JSON_TYPE_MAP.get((spec or {}).get("type"), str)
        fields[prop] = (py_type, ... if prop in required else None)
    return create_model(f"{tool_name}_Args", **fields)


def _wrap_gateway_tool(session: Any, tool_def: Dict[str, Any]) -> Any:
    """Wrap one Action Gateway tool definition as a CrewAI BaseTool.

    Invokes via `session.tools.call` — a direct catalog call (no invoke
    envelope), raising GatewayToolError on failure so CrewAI's own tool
    error handling takes over.
    """
    from crewai.tools import BaseTool

    name = tool_def.get("name")
    description = tool_def.get("description") or f"Action Gateway tool {name}"
    args_model = _args_model(name, tool_def.get("inputSchema") or {})

    tool_kwargs: Dict[str, Any] = {"name": name, "description": description}
    if args_model is not None:
        tool_kwargs["args_schema"] = args_model

    class _GatewayTool(BaseTool):
        def _run(self, **kwargs: Any) -> Any:
            return session.tools.call(name, kwargs)

    return _GatewayTool(**tool_kwargs)


def _load_action_gateway_tools() -> List[Any]:
    """Mint an Action Gateway session via pydo and wrap its tools for CrewAI.

    Best-effort: any failure here (no token, gateway unreachable, tool not
    found) yields no tools rather than failing the whole Crew, matching how
    a customer would defensively wire an optional tool.
    """
    token = os.environ.get("DIGITALOCEAN_TOKEN", "")
    if not token:
        return []

    requested = [
        t.strip()
        for t in os.environ.get("ACTION_GATEWAY_TOOLS", "exa_web_search").split(",")
        if t.strip()
    ]
    if not requested:
        return []

    try:
        from pydo.action_gateway import ActionGatewayClient

        client = ActionGatewayClient(token=token)
        session = client.session.create(
            actor_id=os.environ.get("SESSION_ID", "crewai-ohr-e2e-fixture"),
            tools=requested,
            config={"preloadTools": requested},
            permissions={
                "default_action": "deny",
                "rules": [{"tool": name, "action": "allow"} for name in requested],
            },
        )
    except Exception as exc:  # noqa: BLE001 - defensive, log and continue
        print(f"[e2e_crew] Action Gateway session create failed: {exc}")
        return []

    try:
        catalog = {
            t.get("name"): t
            for t in session.tools.list(include_all=True)
            if t.get("name") in requested
        }
        missing = set(requested) - set(catalog)
        if missing:
            print(f"[e2e_crew] Action Gateway tools not found in session: {sorted(missing)}")
        tools = [_wrap_gateway_tool(session, catalog[name]) for name in catalog]
        print(f"[e2e_crew] Action Gateway tools discovered: {[t.name for t in tools]}")
        return tools
    except Exception as exc:  # noqa: BLE001 - defensive, log and continue
        print(f"[e2e_crew] Action Gateway tool wiring failed: {exc}")
        return []


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

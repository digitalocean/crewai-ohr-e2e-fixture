"""Two-agent sequential CrewAI Crew for OHR kickoff E2E.

Uses a real LLM (OPENAI_API_KEY + MODEL from the session env). One kickoff
runs Researcher → Writer in Process.sequential — exactly the OSS Crew path
the guest mediator drives via `akickoff(inputs=…)`.
"""

from __future__ import annotations

import os

from crewai import Agent, Crew, LLM, Process, Task

_model = os.environ.get("MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o"
_llm = LLM(model=_model, api_key=os.environ.get("OPENAI_API_KEY"))

researcher = Agent(
    role="Researcher",
    goal="Gather 2-3 crisp facts about the given topic",
    backstory="You are a concise researcher. Prefer short bullet facts.",
    llm=_llm,
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

research_task = Task(
    description=(
        "Research this topic and return exactly 2-3 short bullet facts. "
        "Topic: {topic}"
    ),
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

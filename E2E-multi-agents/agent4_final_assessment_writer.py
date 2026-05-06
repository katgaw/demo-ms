"""
Final assessment writer — the narrative report is produced by an LLM.

Flow (high level)
-----------------
1. ``build_final_assessment_writer_agent(llm)`` wraps the chat model in LangChain's ``create_agent``
   with one tool: ``format_borrower_header``.
2. The graph node builds a **HumanMessage** containing ``borrower_id`` + ``rule_results`` (JSON).
   The LLM may call the tool, then writes the full prose assessment.
3. The node's **output** is mostly the model's last reply text, stored as ``final_report``.

What goes *into* the LLM (per invoke)
-------------------------------------
- **System**: instructions in ``build_final_assessment_writer_agent`` (role + report structure).
- **Messages**: prior conversation from graph state (if any), plus one new user message with:
  ``borrower_id``, serialized ``rule_results``, and optionally orchestrator ``writer_feedback``
  + a trimmed prior ``final_report`` when rewriting.

What comes *out* of this module (graph node return)
---------------------------------------------------
- ``final_report``: string — plain assessment text (used by Streamlit / PDF / orchestrator).
- ``draft_complete``: ``True``.
- ``messages``: one ``AIMessage`` tagging the specialist name (notebook-style), same body as ``final_report``.
"""

from __future__ import annotations

import json
from typing import Any

from _lc_compat import get_create_agent

create_agent = get_create_agent()
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def format_borrower_header(borrower_id: int) -> str:
    """Return the standard title line for this borrower's compliance report."""
    return f"Margin loan compliance assessment — borrower_id={borrower_id}"


def build_final_assessment_writer_agent(llm: ChatOpenAI):
    return create_agent(
        model=llm,
        tools=[format_borrower_header],
        system_prompt=(
            "You are the final-assessment-writer specialist. "
            "Start by calling format_borrower_header for the borrower_id provided in the user message. "
            "Then produce a formal report: for each rule, step-by-step reasoning, CONFORMS or DOES NOT CONFORM, and why. "
            "End with an overall conclusion."
        ),
    )


def _revision_addon(writer_feedback: str, prior_report: str) -> str:
    """Extra instructions when the orchestrator asked for a rewrite."""
    if not writer_feedback:
        return ""
    return (
        f"\n\nOrchestrator revision feedback:\n{writer_feedback}\n\n"
        f"Prior draft (trimmed):\n{prior_report[:4000]}\n"
        "Rewrite fully; stay faithful to the JSON outcomes."
    )


def _writer_task(
    borrower_id: int,
    rule_results: Any,
    writer_feedback: str,
    prior_report: str,
) -> HumanMessage:
    """Single user turn: JSON outcomes + optional revision context + what to do next."""
    blob = json.dumps(rule_results or [], indent=2, default=str)
    extra = _revision_addon(writer_feedback, prior_report)
    content = (
        f"borrower_id={borrower_id}. Rule results JSON:\n{blob}\n"
        f"{extra}\n"
        "Call format_borrower_header first, then write the complete assessment."
    )
    return HumanMessage(content=content)


def _last_message_text(message: Any) -> str:
    raw = message.content if hasattr(message, "content") else message
    return str(raw).strip()


def create_agent_node(agent, name: str):
    """Return a LangGraph-style callable: ``state -> partial state update``.

    Reads from ``state`` (inputs to the writer step)
        ``borrower_id`` (required),
        ``rule_results`` (list, JSON-serialized into the user message),
        ``messages`` (optional chat history passed through to ``agent.invoke``),
        ``writer_feedback`` / ``final_report`` (optional; used together for rewrite rounds).

    Writes into the returned dict (outputs)
        ``final_report`` — LLM-authored assessment (last assistant message text),
        ``draft_complete`` — ``True``,
        ``messages`` — tagged ``AIMessage`` for the transcript (see module docstring).
    """

    def agent_node(state: dict[str, Any]) -> dict[str, Any]:
        # 1) Build the prompt from graph state
        task = _writer_task(
            int(state["borrower_id"]),
            state.get("rule_results"),
            (state.get("writer_feedback") or "").strip(),
            (state.get("final_report") or "").strip(),
        )
        # 2) Run the specialist and capture the plain report (for final_report / PDF / orchestrator)
        prior_msgs: list[BaseMessage] = list(state.get("messages") or [])
        out = agent.invoke({"messages": prior_msgs + [task]})
        report_body = _last_message_text(out["messages"][-1])
        # 3) Tag the reply like the assignment notebook (`[SPECIALIST]` blocks)
        tagged = AIMessage(
            content=f"[{name.upper()} SPECIALIST]\n\n{report_body}",
            name=name,
        )
        return {"final_report": report_body, "draft_complete": True, "messages": [tagged]}

    return agent_node


def final_assessment_writer_node(state: dict[str, Any], llm: ChatOpenAI) -> dict[str, Any]:
    agent = build_final_assessment_writer_agent(llm)
    return create_agent_node(agent, "final_assessment_writer")(state)


__all__ = [
    "format_borrower_header",
    "build_final_assessment_writer_agent",
    "create_agent_node",
    "final_assessment_writer_node",
]

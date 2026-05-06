"""
Margin loan multi-agent workflow (LangGraph): supervisor init; rule generation; rule conformance;
on-demand data extraction when rules need missing fields; final narrative.

Named ``margin_graph.py`` (not ``langgraph.py``) so this module does not shadow the installed
``langgraph`` package on ``sys.path``.

Run from repository root: ``python -m margin_graph "Assess borrower_id=11 ..."``.
"""

from __future__ import annotations

import _lc_compat  # noqa: F401 — langchain_core.memory + function_calling compat

import operator
import os
import re
import sys
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent1_data_extractor import data_extractor_node
from agent2_rule_generator import rule_generator_node
from agent3_rule_conformer import rule_conformer_node
from agent4_final_assessment_writer import final_assessment_writer_node

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(BASE, "data", "fake_margin_loan_dataset.csv")
DEFAULT_POLICY = os.path.join(BASE, "data", "margin_loan_policy.md")


def merge_extracted(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if right is None:
        return left or {}
    if left is None:
        left = {}
    return {**left, **right}


class MarginState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    borrower_id: int
    client_csv_path: str
    policy_path: str
    max_rules: int  # if set (>0), conformance runs at most this many generated rules
    rules: list[dict[str, Any]]
    rule_index: int
    extracted: Annotated[dict[str, Any], merge_extracted]
    variables_requested: list[str]
    need_extraction: bool
    conform_done: bool
    rule_results: Annotated[list[dict[str, Any]], operator.add]
    policy_markdown: str
    final_report: str
    draft_complete: bool


def parse_borrower_id(messages: list[BaseMessage]) -> int:
    text = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            text = str(m.content)
            break
    m = re.search(r"borrower(?:_id)?\s*[:=#]?\s*(\d+)", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return int(m.group(1))
    return 1


def supervisor_node(state: MarginState) -> dict[str, Any]:
    borrower_id = parse_borrower_id(state.get("messages") or [])
    csv_path = state.get("client_csv_path") or os.environ.get("MARGIN_CLIENT_CSV", DEFAULT_CSV)
    policy_path = state.get("policy_path") or os.environ.get("MARGIN_POLICY_PATH", DEFAULT_POLICY)
    return {
        "borrower_id": borrower_id,
        "client_csv_path": csv_path,
        "policy_path": policy_path,
        "rules": [],
        "rule_index": 0,
        "extracted": {},
        "variables_requested": [],
        "need_extraction": False,
        "conform_done": False,
        "policy_markdown": "",
        "final_report": "",
        "draft_complete": False,
    }


# LangGraph node callables take only ``state``. Each agent module exposes ``*_node(state, llm, ...)``;
# these functions wire env-configured models and delegate to the real implementations.


def margin_graph_rule_generator(state: MarginState) -> dict[str, Any]:
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    embeddings = OpenAIEmbeddings(model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    out = rule_generator_node({**state, "policy_path": state["policy_path"]}, llm, embeddings)
    cap = state.get("max_rules")
    rules = out.get("rules") or []
    if cap is not None and cap > 0 and len(rules) > cap:
        out = {**out, "rules": rules[:cap]}
    return out


def margin_graph_rule_conformer(state: MarginState) -> dict[str, Any]:
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    return rule_conformer_node(dict(state), llm)


def margin_graph_data_extractor(state: MarginState) -> dict[str, Any]:
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    return data_extractor_node(dict(state), llm)


def margin_graph_final_writer(state: MarginState) -> dict[str, Any]:
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    return final_assessment_writer_node(dict(state), llm)


def route_after_conformer(state: MarginState) -> Literal["data_extractor", "final_assessment_writer", "rule_conformer"]:
    if state.get("need_extraction"):
        return "data_extractor"
    if state.get("conform_done"):
        return "final_assessment_writer"
    return "rule_conformer"


def build_graph() -> StateGraph:
    g = StateGraph(MarginState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("rule_generator", margin_graph_rule_generator)
    g.add_node("rule_conformer", margin_graph_rule_conformer)
    g.add_node("data_extractor", margin_graph_data_extractor)
    g.add_node("final_assessment_writer", margin_graph_final_writer)

    g.add_edge(START, "supervisor")
    g.add_edge("supervisor", "rule_generator")
    g.add_edge("rule_generator", "rule_conformer")

    g.add_conditional_edges(
        "rule_conformer",
        route_after_conformer,
        {
            "data_extractor": "data_extractor",
            "final_assessment_writer": "final_assessment_writer",
            "rule_conformer": "rule_conformer",
        },
    )
    g.add_edge("data_extractor", "rule_conformer")
    g.add_edge("final_assessment_writer", END)

    return g.compile()


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in your environment.", file=sys.stderr)
        sys.exit(1)

    user_q = " ".join(sys.argv[1:]).strip() or "Assess borrower_id=11 for margin loan compliance."
    graph = build_graph()
    result = graph.invoke({"messages": [HumanMessage(content=user_q)]})

    print("\n=== FINAL REPORT ===\n")
    print(result.get("final_report") or "(no report)")


if __name__ == "__main__":
    main()

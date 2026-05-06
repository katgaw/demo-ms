"""
Rule conformer: for each rule, ensure the CSV fields that rule needs are present in ``extracted``,
then call the LLM once with the rule text + client values (structured verdict).

No separate “plan variables” model call — column hints come from ``relevant_variables`` on each rule
(from the rule generator). If that list is empty, we fall back to whatever columns are already in
``extracted``.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

ALLOWED_COLUMNS = """
borrower_id, income, net_worth, liquidity, credit_limit, loan_amount, collateral_value,
margin_percentage, margin_value, ltv_ratio, asset_volatility, market_drop_sensitivity,
interest_rate, payment_history_score, legal_risk_flag
"""


def _allowed_column_names() -> set[str]:
    return {x.strip() for x in ALLOWED_COLUMNS.replace("\n", " ").split(",") if x.strip()}


class RuleAssessment(BaseModel):
    conforms: bool
    reasoning: str = Field(description="Short justification referencing actual values")


def columns_for_rule(rule: dict[str, Any], extracted: dict[str, Any]) -> list[str]:
    """Which CSV columns this step cares about (for missing-field checks and ``used_values``)."""
    allow = _allowed_column_names()
    hinted = [str(v).strip() for v in (rule.get("relevant_variables") or []) if str(v).strip() in allow]
    if hinted:
        return list(dict.fromkeys(hinted))
    return [k for k in extracted if k in allow]


def assess_rule_with_values(llm: ChatOpenAI, rule: dict[str, Any], extracted: dict[str, Any]) -> dict[str, Any]:
    """Single LLM call: put the rule and the full client row in the prompt → conforms + reasoning."""
    focus = columns_for_rule(rule, extracted)
    payload = json.dumps(extracted, indent=2, default=str)
    prompt = f"""You verify whether a margin loan client satisfies this rule.
    Rule ID: {rule.get("rule_id")}
    Rule: {rule.get("statement")}
    Pay special attention to these fields (others in the JSON may still matter if the rule needs them): {json.dumps(focus)}
    Full client row (JSON):
    {payload}
    Decide conforms true/false using standard numeric comparison. If a value required for this rule is null or missing, conforms is false and explain what is missing.
    """
    judge = llm.with_structured_output(RuleAssessment)
    verdict: RuleAssessment = judge.invoke(prompt)
    used = {k: extracted[k] for k in focus if k in extracted}
    return {
        "rule_id": rule.get("rule_id"),
        "statement": rule.get("statement"),
        "conforms": verdict.conforms,
        "reasoning": verdict.reasoning,
        "used_values": used,
    }


def _rule_conformer_inner(name: str, llm: ChatOpenAI):
    """One graph step per rule: missing columns → ask extractor; else LLM assessment."""

    def agent_node(state: dict[str, Any]) -> dict[str, Any]:
        rules: list[dict[str, Any]] = state.get("rules") or []
        idx = int(state.get("rule_index", 0))
        extracted: dict[str, Any] = dict(state.get("extracted") or {})

        if idx >= len(rules):
            return {"conform_done": True, "need_extraction": False, "variables_requested": []}

        rule = rules[idx]
        needed = columns_for_rule(rule, extracted)
        missing = [v for v in needed if v not in extracted or extracted.get(v) is None]
        if missing:
            tagged = AIMessage(
                content=(
                    f"[{name.upper()} SPECIALIST]\n\n"
                    f"Required columns for this rule: {json.dumps(needed)}\n"
                    f"Missing from extracted row: {json.dumps(missing)}"
                ),
                name=name,
            )
            return {
                "variables_requested": missing,
                "need_extraction": True,
                "conform_done": False,
                "messages": [tagged],
            }

        verdict = assess_rule_with_values(llm, rule, extracted)
        tagged = AIMessage(
            content=(
                f"[{name.upper()} SPECIALIST]\n\n"
                f"Columns considered: {json.dumps(needed)}\n"
                f"Conforms: {verdict.get('conforms')}\n"
                f"Reasoning: {verdict.get('reasoning')}"
            ),
            name=name,
        )
        return {
            "rule_results": [verdict],
            "rule_index": idx + 1,
            "need_extraction": False,
            "variables_requested": [],
            "conform_done": idx + 1 >= len(rules),
            "messages": [tagged],
        }

    return agent_node


def create_agent_node(agent: Any, name: str, llm: ChatOpenAI):
    """Return the rule-conformer step callable. *agent* is unused (backward-compatible signature)."""

    return _rule_conformer_inner(name, llm)


def rule_conformer_node(state: dict[str, Any], llm: ChatOpenAI) -> dict[str, Any]:
    return _rule_conformer_inner("rule_conformer", llm)(state)


__all__ = [
    "ALLOWED_COLUMNS",
    "columns_for_rule",
    "assess_rule_with_values",
    "create_agent_node",
    "rule_conformer_node",
]

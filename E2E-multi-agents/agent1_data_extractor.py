"""
Data extractor: borrower fields via langchain_experimental create_pandas_dataframe_agent,
with deterministic CSV row read as fallback (LangChain dataframe-agent pattern).
"""

from __future__ import annotations

import _lc_compat  # noqa: F401 — langchain_core.memory + function_calling compat

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent


def _resolve_column(name: str, columns: list[str]) -> str | None:
    if name in columns:
        return name
    lower_map = {c.lower(): c for c in columns}
    return lower_map.get(name.strip().lower())


def extract_variables_for_borrower(
    borrower_id: int,
    variable_names: list[str],
    csv_path: str | Path,
) -> dict[str, Any]:
    """Deterministic row read (fallback if the pandas agent output is not valid JSON)."""
    path = Path(csv_path).resolve()
    df = pd.read_csv(path)
    if "borrower_id" not in df.columns:
        return {v: None for v in variable_names}

    row_df = df.loc[df["borrower_id"] == borrower_id]
    if row_df.empty:
        return {v: None for v in variable_names}

    row = row_df.iloc[0]
    columns = list(df.columns)
    out: dict[str, Any] = {}
    for raw_name in variable_names:
        col = _resolve_column(raw_name.strip(), columns)
        if col is None:
            out[raw_name] = None
            continue
        val = row[col]
        if pd.isna(val):
            out[col] = None
        else:
            raw = val.item() if hasattr(val, "item") else val
            if isinstance(raw, float) and raw.is_integer():
                out[col] = int(raw)
            else:
                out[col] = raw
    return out


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def build_pandas_dataframe_agent(llm: ChatOpenAI, df: pd.DataFrame):
    """Pandas dataframe agent (requires allow_dangerous_code=True per LangChain security notice)."""
    return create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="tool-calling",
        verbose=False,
        allow_dangerous_code=True,
    )


def create_agent_node(agent, name: str):
    """Notebook-style wrapper: run the dataframe agent and tag the AIMessage."""

    def agent_node(state: dict[str, Any]) -> dict[str, Any]:
        requested = list(state.get("variables_requested") or [])
        if not requested:
            return {"need_extraction": False, "variables_requested": []}

        borrower_id = int(state["borrower_id"])
        cols = ", ".join(requested)
        question = (
            f"The dataframe is client margin loan data. Filter to the single row where borrower_id == {borrower_id}. "
            f"Return ONLY one JSON object whose keys are exactly these column names: {cols}. "
            "Values must be scalars (numbers, strings, or null). No markdown, no explanation outside the JSON."
        )
        result = agent.invoke({"input": question})
        out_text = str(result.get("output", "")).strip()

        new_extracted = _parse_json_object(out_text) or extract_variables_for_borrower(
            borrower_id,
            requested,
            state["client_csv_path"],
        )

        response_with_name = AIMessage(
            content=f"[{name.upper()} SPECIALIST]\n\n{out_text}",
            name=name,
        )
        return {
            "extracted": new_extracted,
            "variables_requested": [],
            "need_extraction": False,
            "messages": [response_with_name],
        }

    return agent_node


def data_extractor_node(state: dict[str, Any], llm: ChatOpenAI) -> dict[str, Any]:
    df = pd.read_csv(state["client_csv_path"])
    agent = build_pandas_dataframe_agent(llm, df)
    return create_agent_node(agent, "data_extractor")(state)


__all__ = [
    "BASE_DIR",
    "extract_variables_for_borrower",
    "build_pandas_dataframe_agent",
    "create_agent_node",
    "data_extractor_node",
]

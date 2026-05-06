"""
Streamlit UI for the margin-loan LangGraph workflow.
Run: streamlit run streamlit_app.py
"""

from __future__ import annotations

import _lc_compat  # noqa: F401 — langchain_core.memory + function_calling compat

import html
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import UUID

import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
load_dotenv()

from margin_graph import DEFAULT_CSV, build_graph  # noqa: E402

_APP_DIR = Path(__file__).resolve().parent
DATA_DIR = _APP_DIR / "data"
COMPANY_LOGO_PATH = DATA_DIR / "image.png"
# Written when the user uploads via Streamlit; removed when the uploader is cleared.
UPLOADED_CLIENT_CSV = DATA_DIR / "uploaded_client.csv"
UPLOADED_POLICY_PDF = DATA_DIR / "uploaded_policy.pdf"


def _final_report_pdf_bytes(title: str, body: str) -> bytes:
    """Build a simple multi-page PDF from plain / markdown-ish report text."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = getSampleStyleSheet()
    story: list[Any] = [
        Paragraph(html.escape(title), styles["Title"]),
        Spacer(1, 18),
    ]
    text = (body or "(no report)").strip()
    for chunk in text.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            story.append(Spacer(1, 6))
            continue
        inner = html.escape(chunk).replace("\n", "<br/>")
        story.append(Paragraph(inner, styles["BodyText"]))
        story.append(Spacer(1, 10))
    doc.build(story)
    return buf.getvalue()


def _truncate_display(text: str, max_len: int = 92) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _conformity_pill_html(conforms: bool | None) -> str:
    if conforms is True:
        inner = "CONFORM"
        bg, fg, border = "#14532d", "#bbf7d0", "#22c55e"
    elif conforms is False:
        inner = "NOT-CONFORM"
        bg, fg, border = "#7f1d1d", "#fecaca", "#ef4444"
    else:
        inner = "UNKNOWN"
        bg, fg, border = "#374151", "#e5e7eb", "#6b7280"
    return (
        '<div style="display:flex;justify-content:center;align-items:flex-start;padding-top:4px;">'
        '<span style="display:inline-block;border-radius:9999px;padding:10px 18px;font-weight:700;'
        f"font-size:0.72rem;letter-spacing:0.06em;white-space:nowrap;background:{bg};color:{fg};"
        f'border:2px solid {border};">{html.escape(inner)}</span></div>'
    )


def _render_rule_outcomes_table(results: list[dict[str, Any]]) -> None:
    """Table-like rows: expandable rule name (left), oval conformity badge (right)."""
    if not results:
        st.info("No structured rule results for this run.")
        return

    st.subheader("Rule outcomes")
    st.caption("Open a rule to see reasoning and values used.")

    hdr_l, hdr_r = st.columns([5, 1])
    with hdr_l:
        st.markdown("**Rule**")
    with hdr_r:
        st.markdown("**Outcome**")

    st.divider()

    for i, r in enumerate(results):
        rid = str(r.get("rule_id", "—"))
        stmt = _truncate_display(str(r.get("statement", "")))
        label = f"{rid}: {stmt}"
        row_l, row_r = st.columns([5, 1])
        with row_l:
            with st.expander(label, expanded=False, key=f"rule_outcome_exp_{i}"):
                reasoning = str(r.get("reasoning") or "").strip() or "_No reasoning._"
                st.markdown(reasoning)
                used = r.get("used_values")
                if used:
                    st.markdown("**Values used**")
                    st.code(json.dumps(used, indent=2, default=str), language="json")
        with row_r:
            st.markdown(_conformity_pill_html(r.get("conforms")), unsafe_allow_html=True)


def _persist_upload_to_data(uploaded: Any | None, dest: Path) -> str | None:
    """Save the active upload under `data/` (overwrite). Delete `dest` when upload is cleared."""
    if uploaded is None:
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        return None
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(uploaded.getbuffer())
    return str(dest.resolve())


NODE_LABELS: dict[str, str] = {
    "supervisor": "Supervisor — load paths & borrower id",
    "rule_generator": "Rule generator — policy → rules + RAG",
    "rule_conformer": "Rule conformer — check fields / LLM assesses rule",
    "data_extractor": "Data extractor — pandas dataframe agent",
    "final_assessment_writer": "Final writer — narrative report",
}


class WorkflowCallbackHandler(BaseCallbackHandler):
    """Captures tool and retriever calls from LangChain agents inside graph nodes."""

    def __init__(self, lines: list[str], max_input_len: int = 220) -> None:
        self.lines = lines
        self.max_input_len = max_input_len

    def _trim(self, text: str) -> str:
        text = text.replace("\n", " ").strip()
        if len(text) > self.max_input_len:
            return text[: self.max_input_len] + "…"
        return text

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        name = serialized.get("name") or serialized.get("id") or "tool"
        detail = self._trim(input_str) if input_str else ""
        msg = f"🔧 **Tool** `{name}`" + (f" — {detail}" if detail else "")
        self.lines.append(msg)

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        name = serialized.get("name") or "retriever"
        self.lines.append(f"🔍 **Retriever** `{name}` — {self._trim(query)}")


def _flatten_stream_chunk(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, dict):
        return chunk
    if isinstance(chunk, (list, tuple)) and len(chunk) == 2 and isinstance(chunk[1], dict):
        return chunk[1]
    return {}


def _render_sidebar_log(log_placeholder: Any, lines: list[str]) -> None:
    body = "\n\n".join(lines) if lines else "_Waiting…_"
    log_placeholder.markdown(body)


def main() -> None:
    st.set_page_config(
        page_title="Margin loan agents",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Set `OPENAI_API_KEY` in `.env` or your environment.")
        return

    header_logo, header_text = st.columns((1, 5))
    with header_logo:
        if COMPANY_LOGO_PATH.is_file():
            st.image(str(COMPANY_LOGO_PATH), width=140)
    with header_text:
        st.markdown("## Margin loan compliance — multi-agent workflow")
        st.caption("LangGraph orchestration with live tool/node activity in the sidebar.")

    with st.sidebar:
        st.header("Activity log")
        st.caption("Full workflow: graph nodes + LangChain tools/retrievers.")
        log_placeholder = st.empty()

    user_q = st.text_area(
        "Request",
        value="Assess borrower_id=11 for margin loan compliance.",
        height=100,
        help="Include borrower_id=… to select the client row.",
    )
    csv_upload = st.file_uploader(
        "structured data (.csv)",
        type=["csv"],
        help=f"Saved to `{UPLOADED_CLIENT_CSV.relative_to(_APP_DIR)}` while selected. "
        "Clear to use `MARGIN_CLIENT_CSV` / default.",
        key="structured_csv_upload",
    )
    pdf_upload = st.file_uploader(
        "rule guide (.pdf)",
        type=["pdf"],
        help=f"Saved to `{UPLOADED_POLICY_PDF.relative_to(_APP_DIR)}` while selected. "
        "Clear to use `MARGIN_POLICY_PATH` / default.",
        key="rule_guide_pdf_upload",
    )

    csv_from_upload = _persist_upload_to_data(csv_upload, UPLOADED_CLIENT_CSV)
    policy_from_upload = _persist_upload_to_data(pdf_upload, UPLOADED_POLICY_PDF)

    if st.button("Run workflow", type="primary", use_container_width=True):
        log_lines: list[str] = []
        handler = WorkflowCallbackHandler(log_lines)
        graph = build_graph()

        initial: dict[str, Any] = {
            "messages": [HumanMessage(content=user_q)],
            "max_rules": 5,
        }
        if csv_from_upload:
            initial["client_csv_path"] = csv_from_upload
        if policy_from_upload:
            initial["policy_path"] = policy_from_upload

        final_state: dict[str, Any] | None = None

        with st.spinner("Running LangGraph…"):
            try:
                for mode, chunk in graph.stream(
                    initial,
                    stream_mode=["updates", "values"],
                    config={"callbacks": [handler]},
                ):
                    if mode == "values":
                        final_state = chunk
                        continue
                    updates = _flatten_stream_chunk(chunk)
                    for node_name, node_update in updates.items():
                        label = NODE_LABELS.get(node_name, f"Node `{node_name}`")
                        log_lines.append(f"**▶ Agent / step:** {label}")
                        if isinstance(node_update, dict) and node_update:
                            keys = ", ".join(sorted(node_update.keys()))
                            log_lines.append(f" ↳ state keys updated: `{keys}`")
                        _render_sidebar_log(log_placeholder, log_lines)
            except Exception as e:
                log_lines.append(f"❌ **Error:** `{e}`")
                _render_sidebar_log(log_placeholder, log_lines)
                st.exception(e)
                return

        _render_sidebar_log(log_placeholder, log_lines)

        if final_state is None:
            st.error("Workflow produced no final state.")
            return

        st.divider()
        st.subheader("Final report")
        report = final_state.get("final_report") or "(no report)"
        st.download_button(
            label="Download report as PDF",
            data=_final_report_pdf_bytes("Margin loan compliance — final report", report),
            file_name="margin_loan_final_report.pdf",
            mime="application/pdf",
            key="download_final_report_pdf",
        )
        st.markdown(report)

        _render_rule_outcomes_table(final_state.get("rule_results") or [])

        with st.expander("Raw rule results (JSON)"):
            st.json(final_state.get("rule_results") or [])


if __name__ == "__main__":
    main()

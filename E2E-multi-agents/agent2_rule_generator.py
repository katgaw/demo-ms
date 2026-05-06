"""
Rule generator: one visible pipeline — policy file → Markdown → chunks → Qdrant vector store →
retriever-backed tool → LangChain agent → structured rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _lc_compat import get_create_agent

create_agent = get_create_agent()
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markitdown import MarkItDown
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = BASE_DIR / "artifacts" / "markdown"

# Optional hook for notebooks / debugging (same retriever reference after a run)
_retriever_holder: dict[str, Any] = {"retriever": None}


class RuleItem(BaseModel):
    rule_id: str = Field(description="Stable identifier e.g. R1, R2")
    statement: str = Field(description="Single logical rule the client must adhere to")
    relevant_variables: list[str] = Field(
        default_factory=list,
        description="CSV column names needed to verify this rule",
    )


class RulesManifest(BaseModel):
    rules: list[RuleItem]


def rule_generator_node(state: dict[str, Any], llm: ChatOpenAI, embeddings: OpenAIEmbeddings) -> dict[str, Any]:
    policy_path = Path(state["policy_path"]).resolve()

    # --- 1. Policy file (e.g. PDF) → Markdown on disk + in memory ---
    DEFAULT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    converted = MarkItDown().convert(str(policy_path))
    policy_markdown = converted.markdown or ""
    md_path = DEFAULT_ARTIFACTS_DIR / f"{policy_path.stem}.md"
    md_path.write_text(f"# Policy source: `{policy_path.name}`\n\n{policy_markdown}\n", encoding="utf-8")

    # --- 2. Markdown → chunks (semantic split if available, else fixed-size) ---
    try:
        from langchain_experimental.text_splitter import SemanticChunker

        chunker_docs = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=75,
        ).create_documents([policy_markdown])
    except Exception:
        chunker_docs = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80).create_documents(
            [policy_markdown]
        )
    chunks: list[Document] = chunker_docs if chunker_docs else [Document(page_content=policy_markdown)]

    # --- 3. Chunks → in-memory Qdrant collection → LangChain retriever ---
    embed_dim = len(embeddings.embed_query("dim"))
    qdrant = QdrantClient(":memory:")
    collection_name = "margin_policy_rules"
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=qdrant,
        collection_name=collection_name,
        embedding=embeddings,
    )
    vector_store.add_documents(chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": min(6, len(chunks) or 1)})
    _retriever_holder["retriever"] = retriever

    # --- 4. Retriever wrapped as a tool + specialist agent ---
    @tool
    def search_margin_policy(query: str) -> str:
        """Search indexed margin loan policy chunks for requirements, thresholds, and eligibility rules."""
        docs = retriever.invoke(query)
        if not docs:
            return "No excerpts found."
        return "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))

    agent = create_agent(
        model=llm,
        tools=[search_margin_policy],
        system_prompt=(
            "You are the rule-generator specialist for margin loans. "
            "Always call search_margin_policy before stating firm requirements. "
            "Cover LTV, maintenance margin, legal risk, payment history, credit limits, and liquidity."
        ),
    )

    task = HumanMessage(
        content=(
            "Use search_margin_policy with diverse queries (LTV, maintenance margin, legal_risk_flag, "
            "payment_history_score, credit_limit, liquidity). Summarize the strongest requirements found."
        )
    )
    base_messages: list[BaseMessage] = list(state.get("messages") or [])
    agent_out = agent.invoke({"messages": base_messages + [task]})
    agent_reply = agent_out["messages"][-1]

    # --- 5. Same retriever + full MD → structured rule list (downstream conformance) ---
    context_docs = retriever.invoke("margin loan eligibility thresholds limits constraints")
    rag_context = (
        "\n\n".join(f"[chunk {i}] {d.page_content}" for i, d in enumerate(context_docs, start=1))
        if context_docs
        else ""
    )
    rules_prompt = f"""You are a compliance analyst for margin lending.
        Full policy (Markdown):
        ---
        {policy_markdown[:14000]}
        ---

        Retrieved excerpts (may overlap):
        ---
        {rag_context[:6000]}
        ---
        Extract EVERY distinct logical rule a borrower must satisfy. Use only these variable names when listing relevant_variables:
        borrower_id, income, net_worth, liquidity, credit_limit, loan_amount, collateral_value,
        margin_percentage, margin_value, ltv_ratio, asset_volatility, market_drop_sensitivity,
        interest_rate, payment_history_score, legal_risk_flag

        Return structured rules with clear statements suitable for automated checking.
        """
    manifest: RulesManifest = llm.with_structured_output(RulesManifest).invoke(rules_prompt)
    rules: list[dict[str, Any]] = [r.model_dump() for r in manifest.rules]

    tagged = AIMessage(
        content=f"[RULE_GENERATOR SPECIALIST]\n\n{agent_reply.content}",
        name="rule_generator",
    )
    return {
        "policy_markdown": policy_markdown,
        "rules": rules,
        "rule_index": 0,
        "messages": [tagged],
    }


__all__ = ["rule_generator_node"]

"""
LangChain / LangChain-Core compatibility layer.

Import this before other LangChain imports when using mixed or legacy environments:

- `langchain_core.memory` was removed → shim onto langchain-classic.
- Legacy `langchain` expects `format_tool_to_openai_function` on
  `langchain_core.utils.function_calling`; newer core only exposes
  `_format_tool_to_openai_function` → alias when missing.

Prefer a single venv with `requirements.txt` (langchain>=1, matching langchain-core).
"""

from __future__ import annotations

import sys
import types
import warnings


def install_langchain_core_memory_shim() -> None:
    root = "langchain_core.memory"
    if root in sys.modules:
        return

    try:
        from langchain_classic.base_memory import BaseMemory
        from langchain_classic.memory.buffer import (
            ConversationBufferMemory,
            ConversationStringBufferMemory,
        )
        from langchain_classic.memory.chat_memory import BaseChatMemory
        from langchain_classic.memory.summary import ConversationSummaryMemory
    except ImportError as e:
        warnings.warn(
            "Could not install langchain_core.memory shim (install langchain-classic): "
            f"{e}",
            stacklevel=2,
        )
        return

    pkg = types.ModuleType(root)
    pkg.__path__ = []
    pkg.BaseMemory = BaseMemory

    chat_mod = types.ModuleType(f"{root}.chat_memory")
    chat_mod.BaseChatMemory = BaseChatMemory

    buf_mod = types.ModuleType(f"{root}.buffer")
    buf_mod.ConversationBufferMemory = ConversationBufferMemory
    buf_mod.ConversationStringBufferMemory = ConversationStringBufferMemory

    summ_mod = types.ModuleType(f"{root}.summary")
    summ_mod.ConversationSummaryMemory = ConversationSummaryMemory

    sys.modules[root] = pkg
    sys.modules[f"{root}.chat_memory"] = chat_mod
    sys.modules[f"{root}.buffer"] = buf_mod
    sys.modules[f"{root}.summary"] = summ_mod


def patch_langchain_core_function_calling() -> None:
    """Restore symbol expected by langchain<1 tool rendering."""
    try:
        import langchain_core.utils.function_calling as fc
    except ImportError:
        return
    if hasattr(fc, "format_tool_to_openai_function"):
        return
    inner = getattr(fc, "_format_tool_to_openai_function", None)
    if inner is not None:
        fc.format_tool_to_openai_function = inner


def warn_if_legacy_langchain_root() -> None:
    try:
        import langchain

        ver = getattr(langchain, "__version__", "") or ""
        parts = ver.split(".")
        major = int(parts[0]) if parts and parts[0].isdigit() else None
        if major is not None and major < 1:
            warnings.warn(
                f"Detected langchain {ver!r}. This repo targets langchain>=1.x. "
                "Upgrade with: pip install -U 'langchain>=1.1' or use the project .venv.",
                stacklevel=2,
            )
    except Exception:
        pass


def install_all_langchain_compat() -> None:
    install_langchain_core_memory_shim()
    patch_langchain_core_function_calling()
    warn_if_legacy_langchain_root()


install_all_langchain_compat()


def get_create_agent():
    """Prefer LangChain 1.x factory API (avoids legacy langchain.agents package graph when possible)."""
    try:
        from langchain.agents.factory import create_agent as ca

        return ca
    except ImportError:
        pass
    try:
        from langchain.agents import create_agent as ca

        return ca
    except ImportError as e:
        raise ImportError(
            "Could not import create_agent. Install LangChain 1.x: pip install -U 'langchain>=1.1'"
        ) from e

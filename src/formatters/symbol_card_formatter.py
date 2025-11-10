"""Formatting utilities for rendering :class:`SymbolCard` instances."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.models.symbol_card import SymbolCard


def format_symbol_card(card: SymbolCard) -> str:
    """Render a human friendly representation of a symbol card."""

    sections = [
        f"[SYMBOL]\n{card.qualified_name} :: {card.signature}\nLocation: {card.location}"
    ]

    doc = card.doc_comment.strip() if card.doc_comment else ""
    if doc:
        sections.append(f"[DOC]\n{doc}")

    if card.callees:
        callees = "\n".join(sorted(card.callees))
        sections.append(f"[CALLEES]\n{callees}")

    snippet = card.snippet.strip()
    if snippet:
        sections.append(f"[SNIPPET]\n{snippet}")

    return "\n\n".join(sections)


def export_metadata(cards: Iterable[SymbolCard], output_path: Path) -> None:
    """Persist metadata for downstream retrieval pipelines."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for card in cards:
            json.dump(card.to_metadata(), handle)
            handle.write("\n")

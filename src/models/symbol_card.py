"""Data structures representing rich symbol information for retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SymbolCard:
    """Serializable representation of a parsed symbol enriched for RAG."""

    usr: str
    name: str
    kind: str
    signature: str
    namespaces: List[str]
    file: str
    line_span: Dict[str, int]
    doc_comment: Optional[str]
    includes: List[str]
    callees: List[str]
    snippet: str
    score: Optional[float] = None
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        """Return the fully qualified name for display."""

        return "::".join([*self.namespaces, self.name]) if self.namespaces else self.name

    @property
    def location(self) -> str:
        start = self.line_span.get("start")
        end = self.line_span.get("end")
        span = f"{start}-{end}" if start and end else "unknown"
        return f"{self.file}:{span}"

    def to_metadata(self) -> Dict[str, object]:
        data = {
            "usr": self.usr,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "kind": self.kind,
            "signature": self.signature,
            "namespaces": self.namespaces,
            "file": self.file,
            "line_span": self.line_span,
            "doc_comment": self.doc_comment,
            "includes": self.includes,
            "callees": self.callees,
        }
        if self.score is not None:
            data["score"] = self.score
        data.update(self.extra)
        return data

    @classmethod
    def from_parsed_symbol(
        cls,
        parsed: "ParsedSymbol",
        snippet: str,
        extra: Optional[Dict[str, object]] = None,
    ) -> "SymbolCard":
        from src.parsers.cpp_parser import ParsedSymbol  # local import to avoid cycle

        if not isinstance(parsed, ParsedSymbol):  # pragma: no cover - defensive
            raise TypeError("parsed must be ParsedSymbol")

        return cls(
            usr=parsed.usr,
            name=parsed.name,
            kind=parsed.kind,
            signature=parsed.signature,
            namespaces=parsed.namespaces,
            file=parsed.file,
            line_span={"start": parsed.line_span[0], "end": parsed.line_span[1]},
            doc_comment=parsed.doc_comment,
            includes=parsed.includes,
            callees=parsed.callees,
            snippet=snippet,
            extra=extra or {},
        )


def build_snippet(file_path: str, start: int, end: int, context: int = 2) -> str:
    """Read the file and return a snippet with the symbol definition."""

    path = Path(file_path)
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    s = max(0, start - context - 1)
    e = min(len(lines), end + context)
    snippet_lines = lines[s:e]
    numbered = [
        f"{i:04d}: {line}" for i, line in enumerate(snippet_lines, start=s + 1)
    ]
    return "\n".join(numbered)

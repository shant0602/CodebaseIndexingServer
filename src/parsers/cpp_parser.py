"""Utilities for parsing C++ projects into structured symbol information.

This module wraps libclang to walk translation units and collect symbols from
project owned source files. The parser intentionally ignores locations that
resolve outside of the provided project root so that headers from system or
third party dependencies are skipped.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    from clang import cindex
except ImportError as exc:  # pragma: no cover - exercised in integration flows
    raise RuntimeError(
        "libclang is required for cpp parsing. Ensure clang bindings are installed"
    ) from exc

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".cpp", ".cc", ".cxx", ".hpp", ".h"}


@dataclass
class ParsedSymbol:
    """Lightweight container describing a symbol extracted from C++ sources."""

    usr: str
    name: str
    kind: str
    namespaces: List[str]
    signature: str
    file: str
    line_span: Tuple[int, int]
    doc_comment: Optional[str]
    includes: List[str]
    callees: List[str] = field(default_factory=list)

    def to_metadata(self) -> Dict[str, object]:
        return {
            "usr": self.usr,
            "name": self.name,
            "kind": self.kind,
            "namespaces": self.namespaces,
            "signature": self.signature,
            "file": self.file,
            "line_span": {
                "start": self.line_span[0],
                "end": self.line_span[1],
            },
            "doc_comment": self.doc_comment,
            "includes": self.includes,
            "callees": self.callees,
        }


class CppParser:
    """Parses C++ projects using libclang and a compile commands database."""

    def __init__(
        self,
        project_root: Path,
        compile_commands: Optional[Path] = None,
        clang_library_path: Optional[Path] = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        if not self.project_root.exists():
            raise FileNotFoundError(f"project root {self.project_root} does not exist")

        if clang_library_path:
            cindex.Config.set_library_path(str(clang_library_path))

        if compile_commands is None:
            compile_commands = self.project_root / "compile_commands.json"

        self.compile_commands_path = Path(compile_commands)
        if not self.compile_commands_path.exists():
            raise FileNotFoundError(
                "compile_commands.json not found. Refer to docs on generating it."
            )

        self._compilation_db = cindex.CompilationDatabase.fromDirectory(
            str(self.compile_commands_path.parent)
        )

    def parse(self) -> Iterator[ParsedSymbol]:
        """Iterate over symbols defined in the project owned translation units."""

        translation_units = self._translation_units()
        for tu_path, tu in translation_units:
            includes = list(self._gather_includes(tu, tu_path))
            LOGGER.debug("Processing translation unit %s", tu_path)
            for cursor in tu.cursor.get_children():
                yield from self._extract_symbols(cursor, [], includes)

    # ------------------------------------------------------------------
    def _translation_units(self) -> Iterator[Tuple[Path, cindex.TranslationUnit]]:
        seen: set[Path] = set()
        for command in self._compilation_db.getAllCompileCommands():
            source = Path(command.filename)
            if source.suffix not in SUPPORTED_EXTENSIONS:
                continue
            source = source.resolve()
            if source in seen:
                continue
            seen.add(source)

            args = list(command.arguments)
            try:
                tu = cindex.TranslationUnit.from_source(
                    str(source), args=args[1:], options=cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES
                )
            except cindex.TranslationUnitLoadError as exc:
                LOGGER.warning("Failed to parse %s: %s", source, exc)
                continue
            yield source, tu

    def _extract_symbols(
        self,
        cursor: cindex.Cursor,
        namespace: List[str],
        includes: List[str],
    ) -> Iterator[ParsedSymbol]:
        if cursor.location.file is None:
            return
        file_path = Path(cursor.location.file.name)
        if not self._is_project_file(file_path):
            return

        new_namespace = namespace
        if cursor.kind in {cindex.CursorKind.NAMESPACE, cindex.CursorKind.CLASS_DECL, cindex.CursorKind.STRUCT_DECL}:
            new_namespace = namespace + [cursor.spelling]

        if cursor.kind in self._symbol_kinds():
            symbol = self._build_symbol(cursor, namespace, includes)
            if symbol:
                yield symbol

        for child in cursor.get_children():
            yield from self._extract_symbols(child, new_namespace, includes)

    def _build_symbol(
        self, cursor: cindex.Cursor, namespace: List[str], includes: List[str]
    ) -> Optional[ParsedSymbol]:
        usr = cursor.get_usr()
        if not usr:
            return None

        extent = cursor.extent
        start = extent.start.line
        end = extent.end.line
        doc_comment = cursor.raw_comment

        signature = self._render_signature(cursor)
        namespaces = [n for n in namespace if n]
        name = cursor.spelling or cursor.displayname or usr
        callees = sorted(set(self._collect_callees(cursor)))

        return ParsedSymbol(
            usr=usr,
            name=name,
            kind=cursor.kind.name,
            namespaces=namespaces,
            signature=signature,
            file=str(Path(cursor.location.file.name).resolve()),
            line_span=(start, end),
            doc_comment=doc_comment,
            includes=includes,
            callees=callees,
        )

    def _collect_callees(self, cursor: cindex.Cursor) -> Iterable[str]:
        for child in cursor.get_children():
            if child.kind == cindex.CursorKind.CALL_EXPR:
                referenced = child.get_reference()
                if referenced is not None:
                    usr = referenced.get_usr()
                    if usr:
                        yield usr
            yield from self._collect_callees(child)

    def _render_signature(self, cursor: cindex.Cursor) -> str:
        result_type = cursor.result_type.spelling if cursor.result_type else ""
        args: List[str] = []
        for arg in cursor.get_arguments() if cursor.kind.is_declaration() else []:
            args.append(f"{arg.type.spelling} {arg.spelling}".strip())
        if result_type:
            signature = f"{result_type} {cursor.displayname}({', '.join(args)})"
        else:
            signature = f"{cursor.displayname}({', '.join(args)})"
        return signature.strip()

    def _gather_includes(
        self, tu: cindex.TranslationUnit, tu_path: Path
    ) -> Iterator[str]:
        for include in tu.get_includes():
            header = Path(include.include.name)
            if self._is_project_file(header):
                yield str(header.resolve())

    def _is_project_file(self, file_path: Path) -> bool:
        try:
            file_path = file_path.resolve()
        except FileNotFoundError:
            return False
        return str(file_path).startswith(str(self.project_root))

    @staticmethod
    def _symbol_kinds() -> set:
        return {
            cindex.CursorKind.FUNCTION_DECL,
            cindex.CursorKind.CXX_METHOD,
            cindex.CursorKind.CONSTRUCTOR,
            cindex.CursorKind.DESTRUCTOR,
            cindex.CursorKind.FUNCTION_TEMPLATE,
            cindex.CursorKind.CLASS_DECL,
            cindex.CursorKind.STRUCT_DECL,
        }


def dump_symbols_to_json(symbols: Sequence[ParsedSymbol], output_path: Path) -> None:
    """Helper to persist parsed symbols to disk for further processing."""

    with output_path.open("w", encoding="utf-8") as f:
        for symbol in symbols:
            json.dump(symbol.to_metadata(), f)
            f.write("\n")

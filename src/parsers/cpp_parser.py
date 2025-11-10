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
# Directories we intentionally skip (third-party deps vendored into the tree).
EXCLUDED_DIR_NAMES = {"deps", "_deps"}
# Flags that older libclang/clang versions may not understand. We strip them so
# parsing can continue instead of bailing out or turning warnings into errors.
UNSUPPORTED_WARNING_FLAGS = {"-Wno-dangling-reference"}
# Flags that promote warnings to errors, causing libclang to abort parsing.
DISABLED_ERROR_PROMOTION_FLAGS = {"-Werror", "-pedantic-errors"}
DISABLED_ERROR_PROMOTION_PREFIXES = ("-Werror=",)
# Flags that require a path/file argument we do not want clang to touch (write).
OUTPUT_FLAG_PREFIXES = ("-o", "-MF", "-MT", "-MQ")
OUTPUT_FLAGS_WITH_VALUE = {"-o", "-MF", "-MT", "-MQ"}
# Additional args that make clang behave like a syntax-only checker.
EXTRA_CLANG_ARGS = ("-fsyntax-only",)
# Container mounts where the project might live when compile_commands was generated.
CONTAINER_PROJECT_PREFIXES = (
    Path("/data/project"),
    Path("/workspace/project"),
    Path("/workspace"),
)


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
            clang_path = Path(clang_library_path)
            # If it's a file, use set_library_file, otherwise use set_library_path
            if clang_path.is_file():
                cindex.Config.set_library_file(str(clang_path))
            else:
                cindex.Config.set_library_path(str(clang_path))

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
    def _remap_path(self, path: Path) -> Path:
        """Remap a path from host machine to container if needed."""
        try:
            resolved = path.resolve()
            if resolved.exists():
                return resolved
        except (FileNotFoundError, OSError):
            pass
        
        # If path doesn't exist, try remapping from host paths to container
        path_str = str(path)
        path_obj = Path(path_str)
        
        # Try to remap between host and container paths in either direction.
        # Host -> container (e.g., /Users/... -> /data/project/...)
        if path_str.startswith("/Users/") or path_str.startswith("/home/"):
            parts = path_obj.parts
            project_names = ["habitat-sim", "src"]
            for project_name in project_names:
                for i, part in enumerate(parts):
                    if part != project_name:
                        continue
                    if project_name == "src":
                        rel_parts = parts[i:]
                        remapped = self.project_root / Path(*rel_parts)
                        if remapped.exists():
                            return remapped
                    else:
                        rel_parts = parts[i + 1 :]
                        if rel_parts:
                            remapped = self.project_root / Path(*rel_parts)
                            if remapped.exists():
                                return remapped
                    break

        # Container -> host (e.g., /data/project/... -> /Users/.../repo)
        for prefix in CONTAINER_PROJECT_PREFIXES:
            try:
                rel = path_obj.relative_to(prefix)
            except ValueError:
                continue
            remapped = self.project_root / rel
            if remapped.exists():
                return remapped
            # Even if the file is generated later, return the remapped path so callers
            # can decide whether to keep or skip it.
            return remapped
        
        # If still not found, try as relative to project root
        if not path.is_absolute():
            remapped = self.project_root / path
            if remapped.exists():
                return remapped
        
        return path

    def _translation_units(self) -> Iterator[Tuple[Path, cindex.TranslationUnit]]:
        seen: set[Path] = set()
        total_commands = 0
        skipped_extensions = 0
        skipped_missing = 0
        skipped_excluded = 0
        parsed_successfully = 0
        
        commands = list(self._compilation_db.getAllCompileCommands())
        total_commands = len(commands)
        for idx, command in enumerate(commands, start=1):
            source = Path(command.filename)
            if source.suffix not in SUPPORTED_EXTENSIONS:
                skipped_extensions += 1
                continue
            
            # Skip vendored dependency trees (e.g., src/deps/**)
            try:
                rel = source.resolve().relative_to(self.project_root)
                if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
                    skipped_excluded += 1
                    continue
            except (ValueError, FileNotFoundError):
                pass

            # Remap path if it doesn't exist (e.g., from host to container)
            original_source = source
            source = self._remap_path(source)
            
            if not source.exists():
                skipped_missing += 1
                if skipped_missing <= 5:  # Log first few for debugging
                    LOGGER.debug("Skipping non-existent file: %s (original: %s)", source, original_source)
                continue
            
            source = source.resolve()
            if source in seen:
                continue
            seen.add(source)

            def _path_str(p: Path) -> str:
                try:
                    return str(p.resolve())
                except FileNotFoundError:
                    return str(p)

            source_variants = {
                str(original_source),
                _path_str(original_source),
                str(source),
                str(source.resolve()),
            }

            # Remap paths in compile arguments as well
            raw_args = list(command.arguments)[1:]  # Skip the compiler binary
            remapped_args: List[str] = []
            i = 0
            while i < len(raw_args):
                arg = raw_args[i]
                i += 1
                if arg in source_variants:
                    LOGGER.debug("Dropping duplicate source arg: %s", arg)
                    continue
                if not arg.startswith("-") and not arg.startswith("@"):
                    try:
                        if Path(arg).resolve() == source:
                            LOGGER.debug("Dropping duplicate source arg: %s", arg)
                            continue
                    except (OSError, ValueError):
                        pass

                if arg in UNSUPPORTED_WARNING_FLAGS:
                    LOGGER.debug("Dropping unsupported compiler flag: %s", arg)
                    continue
                if arg in DISABLED_ERROR_PROMOTION_FLAGS or any(
                    arg.startswith(prefix) for prefix in DISABLED_ERROR_PROMOTION_PREFIXES
                ):
                    LOGGER.debug("Dropping error-promotion flag: %s", arg)
                    continue

                # Drop flags that would force clang to write output artifacts.
                if arg in OUTPUT_FLAGS_WITH_VALUE:
                    if i < len(raw_args):
                        skipped = raw_args[i]
                        i += 1
                        LOGGER.debug("Dropping %s %s to avoid filesystem writes", arg, skipped)
                    else:
                        LOGGER.debug("Dropping dangling flag without value: %s", arg)
                    continue
                if any(arg.startswith(prefix) and len(arg) > len(prefix) for prefix in OUTPUT_FLAG_PREFIXES):
                    LOGGER.debug("Dropping output-producing flag: %s", arg)
                    continue

                # Remap include paths and other file paths in arguments
                if arg.startswith("-I"):
                    include_path_str: Optional[str] = None
                    if len(arg) > 2:
                        include_path_str = arg[2:]
                    elif i < len(raw_args):
                        include_path_str = raw_args[i]
                        i += 1
                    if include_path_str:
                        include_path = Path(include_path_str)
                        if not include_path.exists():
                            remapped_include = self._remap_path(include_path)
                            if remapped_include.exists():
                                remapped_args.append(f"-I{remapped_include}")
                                continue
                            if "/build/" in str(include_path):
                                continue
                        if include_path.exists() or not include_path.is_absolute():
                            remapped_args.append(f"-I{include_path}")
                    continue

                if arg.startswith("-isystem"):
                    include_path_str: Optional[str] = None
                    if len(arg) > 9:
                        include_path_str = arg[9:]
                    elif i < len(raw_args):
                        include_path_str = raw_args[i]
                        i += 1
                    if include_path_str:
                        include_path = Path(include_path_str)
                        if not include_path.exists():
                            remapped_include = self._remap_path(include_path)
                            if remapped_include.exists():
                                remapped_args.append(f"-isystem{remapped_include}")
                                continue
                            if "/build/" in str(include_path):
                                continue
                        if include_path.exists() or not include_path.is_absolute():
                            remapped_args.append(f"-isystem{include_path}")
                    continue

                if arg.startswith("/"):
                    path_arg = Path(arg)
                    if path_arg.exists():
                        remapped_args.append(arg)
                        continue
                    remapped_path = self._remap_path(path_arg)
                    if remapped_path.exists():
                        remapped_args.append(str(remapped_path))
                        continue
                    if "/build/" in str(path_arg):
                        continue
                    remapped_args.append(arg)
                    continue

                remapped_args.append(arg)

            for extra_arg in EXTRA_CLANG_ARGS:
                if extra_arg not in remapped_args:
                    remapped_args.append(extra_arg)

            try:
                tu = cindex.TranslationUnit.from_source(
                    str(source), args=remapped_args, options=cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES
                )
                parsed_successfully += 1
            except cindex.TranslationUnitLoadError as exc:
                error_msg = str(exc)
                # Log a sample of failed files with more context
                if parsed_successfully == 0 and skipped_missing < 3:
                    LOGGER.warning("Failed to parse %s: %s (sample args: %s)", source, error_msg, remapped_args[:5])
                else:
                    LOGGER.warning("Failed to parse %s: %s", source, error_msg)
                continue
            yield source, tu
            if idx % 25 == 0 or idx == total_commands:
                LOGGER.info(
                    "Processed %d/%d commands (parsed=%d, skipped_extensions=%d, skipped_missing=%d, skipped_excluded=%d)",
                    idx,
                    total_commands,
                    parsed_successfully,
                    skipped_extensions,
                    skipped_missing,
                    skipped_excluded,
                )

        LOGGER.info(
            "Translation unit processing summary: total=%d, skipped_extensions=%d, skipped_missing=%d, skipped_excluded=%d, parsed_successfully=%d",
            total_commands, skipped_extensions, skipped_missing, skipped_excluded, parsed_successfully
        )

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
                referenced = None
                get_reference = getattr(child, "get_reference", None)
                if callable(get_reference):
                    try:
                        referenced = get_reference()
                    except (cindex.LibclangError, AttributeError):
                        referenced = None
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
            # Try remapping the path
            file_path = self._remap_path(file_path)
            if not file_path.exists():
                return False
        # Also try remapping if the path doesn't start with project root
        if not str(file_path).startswith(str(self.project_root)):
            remapped = self._remap_path(file_path)
            if remapped.exists() and str(remapped).startswith(str(self.project_root)):
                return True
            return False
        return True

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

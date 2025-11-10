"""CLI for parsing a C++ project and persisting an embedding index."""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence

from src.embeddings.embedder import LocalEmbedder
from src.models.symbol_card import SymbolCard, build_snippet
from src.parsers.cpp_parser import CppParser, ParsedSymbol

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse a C++ project and persist an embedding index for retrieval.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Path to the C++ project root containing compile_commands.json.",
    )
    parser.add_argument(
        "--compile-commands",
        type=Path,
        help="Optional path to compile_commands.json (defaults to project root).",
    )
    parser.add_argument(
        "--generate-compile-commands",
        action="store_true",
        help="Automatically generate compile_commands.json using CMake if it doesn't exist (default: True).",
    )
    parser.add_argument(
        "--no-generate-compile-commands",
        dest="generate_compile_commands",
        action="store_false",
        help="Do not generate compile_commands.json automatically. Fail if it doesn't exist.",
    )
    # Set default to True for auto-generation
    parser.set_defaults(generate_compile_commands=True)
    parser.add_argument(
        "--clang-library-path",
        type=Path,
        help="Optional path to the libclang shared library directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the FAISS index and metadata should be written.",
    )
    parser.add_argument(
        "--index-name",
        default="cpp-symbols",
        help="Name to use for the persisted FAISS index file (without extension).",
    )
    parser.add_argument(
        "--metadata-filename",
        default="metadata.jsonl",
        help="Filename for the metadata JSONL written alongside the index.",
    )
    parser.add_argument(
        "--model-name",
        help="Override the embedding model used by LocalEmbedder.",
    )
    parser.add_argument(
        "--device",
        help="Force the device for embedding computations (e.g., cpu or cuda).",
    )
    parser.add_argument(
        "--context-lines",
        type=int,
        default=2,
        help="Number of context lines to include above and below each symbol snippet.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def generate_compile_commands(project_root: Path, output_path: Path) -> bool:
    """Generate compile_commands.json using CMake.
    
    Returns True if successful, False otherwise.
    """
    if not shutil.which("cmake"):
        LOGGER.error("CMake is not available. Cannot generate compile_commands.json")
        return False
    
    project_root = project_root.resolve()
    build_dir = project_root / "build"
    
    # Check if CMakeLists.txt exists in project root or src subdirectory
    cmake_lists = project_root / "CMakeLists.txt"
    if not cmake_lists.exists():
        cmake_lists = project_root / "src" / "CMakeLists.txt"
        if not cmake_lists.exists():
            LOGGER.error("No CMakeLists.txt found in %s or %s/src", project_root, project_root)
            return False
    
    LOGGER.info("Generating compile_commands.json using CMake...")
    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("CMakeLists.txt: %s", cmake_lists)
    
    try:
        # Clean build directory if it exists (to remove old CMakeCache.txt from host)
        if build_dir.exists():
            LOGGER.info("Removing existing build directory to ensure clean CMake configuration...")
            shutil.rmtree(build_dir)
        
        # Create fresh build directory
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure CMake with compile commands export
        # Use options to make configuration more lenient - we only need compile_commands.json
        source_dir = cmake_lists.parent
        configure_cmd = [
            "cmake",
            "-S", str(source_dir),
            "-B", str(build_dir),
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            # Try to skip optional dependencies that might not be available
            "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=OFF",
            # Allow configuration to continue even with missing packages
            "-Wno-dev",
        ]
        
        # Try to set environment variables to help find packages or skip them
        env = os.environ.copy()
        # Set a flag to indicate we're just generating compile commands
        env["CMAKE_EXPORT_COMPILE_COMMANDS_ONLY"] = "1"
        
        LOGGER.info("Running: %s", " ".join(configure_cmd))
        result = subprocess.run(
            configure_cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env,
        )
        
        # Check if compile_commands.json was generated even if CMake had errors
        compile_commands_src = build_dir / "compile_commands.json"
        if compile_commands_src.exists() and compile_commands_src.stat().st_size > 0:
            LOGGER.info("compile_commands.json was generated despite CMake warnings/errors")
            # Copy it and return success
            LOGGER.info("Copying compile_commands.json to %s", output_path)
            shutil.copy2(compile_commands_src, output_path)
            LOGGER.info("Successfully generated compile_commands.json")
            return True
        
        if result.returncode != 0:
            LOGGER.error("CMake configuration failed:")
            LOGGER.error("stdout: %s", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)  # Last 2000 chars
            LOGGER.error("stderr: %s", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)  # Last 2000 chars
            return False
        
        # Find compile_commands.json in build directory
        compile_commands_src = build_dir / "compile_commands.json"
        if not compile_commands_src.exists():
            LOGGER.error("compile_commands.json not found in build directory: %s", build_dir)
            return False
        
        # Copy to output location
        LOGGER.info("Copying compile_commands.json to %s", output_path)
        shutil.copy2(compile_commands_src, output_path)
        
        LOGGER.info("Successfully generated compile_commands.json")
        return True
        
    except subprocess.TimeoutExpired:
        LOGGER.error("CMake configuration timed out after 5 minutes")
        return False
    except Exception as exc:
        LOGGER.error("Failed to generate compile_commands.json: %s", exc, exc_info=True)
        return False


def collect_symbol_cards(
    symbols: Iterable[ParsedSymbol],
    context_lines: int,
) -> List[SymbolCard]:
    cards: List[SymbolCard] = []
    for symbol in symbols:
        snippet = build_snippet(
            symbol.file,
            symbol.line_span[0],
            symbol.line_span[1],
            context=context_lines,
        )
        cards.append(SymbolCard.from_parsed_symbol(symbol, snippet=snippet))
    return cards


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine compile_commands.json path
    if args.compile_commands:
        compile_commands_path = Path(args.compile_commands)
    else:
        compile_commands_path = args.project_root / "compile_commands.json"
    
    # Check if compile_commands.json exists and if it has valid container paths
    needs_regeneration = False
    if compile_commands_path.exists():
        # Check if the file contains host paths that don't exist in container
        try:
            import json
            with open(compile_commands_path, 'r') as f:
                content = f.read()
                # Check for common host path patterns that won't work in container
                if '/Users/' in content or '/home/' in content:
                    # Check if referenced build directories exist
                    if '/build/' in content:
                        build_paths = [line for line in content.split('\n') if '/build/' in line]
                        # Sample a few paths to see if they exist
                        import re
                        for line in build_paths[:5]:  # Check first 5 build path references
                            matches = re.findall(r'["\']([^"\']*build[^"\']*)["\']', line)
                            for match in matches:
                                if match.startswith('/') and not Path(match).exists():
                                    needs_regeneration = True
                                    LOGGER.info("Found non-existent build paths in compile_commands.json. Will regenerate.")
                                    break
                            if needs_regeneration:
                                break
        except Exception as exc:
            LOGGER.warning("Could not check compile_commands.json validity: %s", exc)
    
    # Generate compile_commands.json if it doesn't exist or needs regeneration
    if not compile_commands_path.exists() or needs_regeneration:
        if args.generate_compile_commands:
            if needs_regeneration:
                LOGGER.info("compile_commands.json contains invalid paths. Regenerating it...")
                # Backup old file
                backup_path = compile_commands_path.with_suffix('.json.bak')
                if compile_commands_path.exists():
                    shutil.move(str(compile_commands_path), str(backup_path))
                    LOGGER.info("Backed up old compile_commands.json to %s", backup_path)
            else:
                LOGGER.info("compile_commands.json not found. Generating it automatically...")
            if not generate_compile_commands(args.project_root, compile_commands_path):
                LOGGER.error("Failed to generate compile_commands.json")
                return 1
        else:
            LOGGER.error(
                "compile_commands.json not found or invalid at %s. "
                "Use --generate-compile-commands (default) to generate it automatically, "
                "or generate it manually using CMake.",
                compile_commands_path
            )
            return 1

    # Use --clang-library-path if provided, otherwise try LIBCLANG_PATH env var
    clang_library_path = args.clang_library_path
    if clang_library_path is None:
        libclang_env = os.environ.get("LIBCLANG_PATH")
        if libclang_env:
            clang_library_path = Path(libclang_env)
            # If it's a file, use it directly; if it's a directory, keep it as directory
            # Parser will handle both cases

    parser = CppParser(
        project_root=args.project_root,
        compile_commands=compile_commands_path,
        clang_library_path=clang_library_path,
    )

    LOGGER.info("Parsing translation units under %s", args.project_root)
    symbols = list(parser.parse())
    if not symbols:
        LOGGER.warning("No symbols discovered. Exiting without writing an index.")
        return 0

    LOGGER.info("Collected %s symbols. Building symbol cards.", len(symbols))
    cards = collect_symbol_cards(symbols, args.context_lines)

    embedder_kwargs = {"index_dir": output_dir, "device": args.device}
    if args.model_name:
        embedder_kwargs["model_name"] = args.model_name

    embedder = LocalEmbedder(**embedder_kwargs)
    LOGGER.info(
        "Embedding %s symbol cards with model %s",
        len(cards),
        embedder.model_name,
    )
    index_path = embedder.index_symbol_cards(
        cards,
        index_name=args.index_name,
        metadata_filename=args.metadata_filename,
    )

    LOGGER.info("Index written to %s", index_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

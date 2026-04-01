import os
from dataclasses import dataclass
from typing import List, Optional
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
import tree_sitter_java as tsjava
import tree_sitter_go as tsgo

from backend.ingestion.walker import FileRecord

@dataclass
class Chunk:
    content: str           # the actual code/text
    file_path: str         # relative path in repo
    language: str          # python, javascript, etc.
    chunk_type: str        # "function", "class", "block", "doc"
    name: str              # function/class name if available
    start_line: int
    end_line: int
    metadata: dict         # extra info for retrieval

# Tree-sitter language registry
LANGUAGE_MAP = {
    "python":     tspython.language(),
    "javascript": tsjavascript.language(),
    "typescript": tstypescript.language_typescript(),
    "java":       tsjava.language(),
    "go":         tsgo.language(),
}

# Node types that represent meaningful code units per language
CHUNK_NODE_TYPES = {
    "python":     ["function_definition", "class_definition"],
    "javascript": ["function_declaration", "class_declaration",
                   "arrow_function", "method_definition"],
    "typescript": ["function_declaration", "class_declaration",
                   "arrow_function", "method_definition"],
    "java":       ["method_declaration", "class_declaration"],
    "go":         ["function_declaration", "method_declaration"],
}

class ASTChunker:
    def __init__(self, max_chunk_lines: int = 80, fallback_chunk_lines: int = 40):
        self.max_chunk_lines = max_chunk_lines
        self.fallback_chunk_lines = fallback_chunk_lines
        self._parsers = {}

    def _get_parser(self, language: str) -> Optional[Parser]:
        """Get or create a tree-sitter parser for a language."""
        if language not in LANGUAGE_MAP:
            return None

        if language not in self._parsers:
            parser = Parser(Language(LANGUAGE_MAP[language]))
            self._parsers[language] = parser

        return self._parsers[language]

    def chunk(self, file_record: FileRecord) -> List[Chunk]:
        """Chunk a file into meaningful units."""
        if file_record.file_type == "doc":
            return self._chunk_doc(file_record)

        parser = self._get_parser(file_record.language)

        if parser:
            chunks = self._chunk_with_ast(file_record, parser)
            if chunks:
                return chunks

        # Fallback: chunk by line blocks
        return self._chunk_by_lines(file_record)

    def _chunk_with_ast(self, file_record: FileRecord, parser: Parser) -> List[Chunk]:
        """Use AST to chunk code by functions and classes."""
        chunks = []
        source = file_record.content.encode("utf-8")
        tree = parser.parse(source)

        node_types = CHUNK_NODE_TYPES.get(file_record.language, [])
        lines = file_record.content.split("\n")

        def extract_name(node) -> str:
            """Extract function or class name from AST node."""
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8")
            return "unknown"

        def visit(node):
            """Recursively visit AST nodes and extract chunks."""
            if node.type in node_types:
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                chunk_lines = end_line - start_line

                # If chunk is too large, recurse into its children
                if chunk_lines > self.max_chunk_lines:
                    for child in node.children:
                        visit(child)
                    return

                content = "\n".join(lines[start_line:end_line + 1])
                name = extract_name(node)

                chunk_type = "function"
                if "class" in node.type:
                    chunk_type = "class"

                chunks.append(Chunk(
                    content=content,
                    file_path=file_record.relative_path,
                    language=file_record.language,
                    chunk_type=chunk_type,
                    name=name,
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                    metadata={
                        "file_path": file_record.relative_path,
                        "language": file_record.language,
                        "chunk_type": chunk_type,
                        "name": name,
                        "start_line": start_line + 1,
                        "end_line": end_line + 1,
                    }
                ))
            else:
                for child in node.children:
                    visit(child)

        visit(tree.root_node)
        return chunks

    def _chunk_by_lines(self, file_record: FileRecord) -> List[Chunk]:
        """Fallback: split file into fixed line blocks."""
        chunks = []
        lines = file_record.content.split("\n")
        step = self.fallback_chunk_lines

        for i in range(0, len(lines), step):
            block = lines[i:i + step]
            content = "\n".join(block).strip()
            if not content:
                continue

            chunks.append(Chunk(
                content=content,
                file_path=file_record.relative_path,
                language=file_record.language,
                chunk_type="block",
                name=f"block_{i}",
                start_line=i + 1,
                end_line=min(i + step, len(lines)),
                metadata={
                    "file_path": file_record.relative_path,
                    "language": file_record.language,
                    "chunk_type": "block",
                    "name": f"block_{i}",
                    "start_line": i + 1,
                    "end_line": min(i + step, len(lines)),
                }
            ))

        return chunks

    def _chunk_doc(self, file_record: FileRecord) -> List[Chunk]:
        """Split documentation files by paragraphs."""
        chunks = []
        paragraphs = file_record.content.split("\n\n")
        current_block = []
        current_line = 1

        for para in paragraphs:
            current_block.append(para)
            block_text = "\n\n".join(current_block)

            # Flush when block gets large enough
            if len(block_text.split("\n")) >= self.fallback_chunk_lines:
                chunks.append(Chunk(
                    content=block_text.strip(),
                    file_path=file_record.relative_path,
                    language="markdown",
                    chunk_type="doc",
                    name=file_record.relative_path,
                    start_line=current_line,
                    end_line=current_line + len(block_text.split("\n")),
                    metadata={
                        "file_path": file_record.relative_path,
                        "language": "markdown",
                        "chunk_type": "doc",
                        "name": file_record.relative_path,
                    }
                ))
                current_line += len(block_text.split("\n"))
                current_block = []

        # Flush remaining content
        if current_block:
            block_text = "\n\n".join(current_block).strip()
            if block_text:
                chunks.append(Chunk(
                    content=block_text,
                    file_path=file_record.relative_path,
                    language="markdown",
                    chunk_type="doc",
                    name=file_record.relative_path,
                    start_line=current_line,
                    end_line=current_line + len(block_text.split("\n")),
                    metadata={
                        "file_path": file_record.relative_path,
                        "language": "markdown",
                        "chunk_type": "doc",
                        "name": file_record.relative_path,
                    }
                ))

        return chunks

    def chunk_many(self, file_records: List[FileRecord]) -> List[Chunk]:
        """Chunk a list of files and return all chunks."""
        all_chunks = []
        for record in file_records:
            chunks = self.chunk(record)
            all_chunks.extend(chunks)

        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
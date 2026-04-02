import os
from dataclasses import dataclass
from typing import List

# File extensions we care about
CODE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
}

DOC_EXTENSIONS = {
    ".md", ".txt", ".rst", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".env.example"
}

# Directories to always skip
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv",
    "venv", "dist", "build", ".next", "coverage",
    ".pytest_cache", ".mypy_cache", "vendor",
    # Skip translated docs — they duplicate content and hurt retrieval
    "docs/zh", "docs/zh-hant", "docs/ko", "docs/pt",
    "docs/ja", "docs/de", "docs/fr", "docs/es",
    "docs/ru", "docs/uk", "docs/tr", "docs/it"
}

@dataclass
class FileRecord:
    path: str           # absolute path
    relative_path: str  # path relative to repo root
    language: str       # "python", "javascript", "doc", etc.
    file_type: str      # "code" or "doc"
    content: str        # raw file content
    size_bytes: int

class FileWalker:
    def __init__(self, max_file_size_kb: int = 500):
        # Skip files larger than this (minified files, etc.)
        self.max_file_size = max_file_size_kb * 1024

    def walk(self, repo_path: str) -> List[FileRecord]:
        """Walk a repo and return all relevant files."""
        records = []

        for root, dirs, files in os.walk(repo_path):
            # Skip irrelevant directories in-place
            dirs[:] = [
                d for d in dirs
                if d not in SKIP_DIRS and not d.startswith(".")
            ]

            for filename in files:
                abs_path = os.path.join(root, filename)
                rel_path = os.path.relpath(abs_path, repo_path)
                ext = os.path.splitext(filename)[1].lower()

                # Determine file type
                if ext in CODE_EXTENSIONS:
                    file_type = "code"
                    language = CODE_EXTENSIONS[ext]
                elif ext in DOC_EXTENSIONS:
                    file_type = "doc"
                    language = "markdown"
                else:
                    continue  # skip unknown file types
                # Skip translated documentation (non-English)
                # Keep only docs/en/ and root-level docs
                if file_type == "doc" and "docs" + os.sep in rel_path:
                    parts = rel_path.split(os.sep)
                    if "docs" in parts:
                        docs_idx = parts.index("docs")
                        # Check if next part is a non-English language code
                        if len(parts) > docs_idx + 1:
                            lang = parts[docs_idx + 1]
                            if lang != "en" and len(lang) <= 7:
                                continue  # skip non-English docs

                # Skip large files
                size = os.path.getsize(abs_path)
                if size > self.max_file_size:
                    print(f"Skipping large file: {rel_path} ({size//1024}KB)")
                    continue

                # Read content
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception as e:
                    print(f"Could not read {rel_path}: {e}")
                    continue

                # Skip empty files
                if not content.strip():
                    continue

                records.append(FileRecord(
                    path=abs_path,
                    relative_path=rel_path,
                    language=language,
                    file_type=file_type,
                    content=content,
                    size_bytes=size
                ))

        print(f"Found {len(records)} files "
              f"({sum(1 for r in records if r.file_type == 'code')} code, "
              f"{sum(1 for r in records if r.file_type == 'doc')} docs)")

        return records
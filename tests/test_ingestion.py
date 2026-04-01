from backend.ingestion.cloner import RepoCloner
from backend.ingestion.walker import FileWalker
from backend.ingestion.chunker import ASTChunker

def test_all():
    cloner = RepoCloner()
    walker = FileWalker()
    chunker = ASTChunker()

    # Clone once, reuse for all tests
    repo_path = cloner.clone("https://github.com/tiangolo/fastapi")
    files = walker.walk(repo_path)

    print(f"\nTotal files: {len(files)}")
    for f in files[:5]:
        print(f"  {f.relative_path} ({f.language})")

    # Grab first 10 code files specifically
    code_files = [f for f in files if f.file_type == "code"][:10]
    doc_files = [f for f in files if f.file_type == "doc"][:5]

    print(f"\nTesting chunker on {len(code_files)} code files + {len(doc_files)} doc files...")
    chunks = chunker.chunk_many(code_files + doc_files)

    print(f"\nSample chunks:")
    for c in chunks[:8]:
        print(f"  [{c.chunk_type}] {c.name} — {c.file_path} "
              f"(lines {c.start_line}-{c.end_line})")

    # Cleanup once at the end
    cloner.cleanup(repo_path)

if __name__ == "__main__":
    test_all()

from backend.ingestion.embedder import Embedder

def test_all():
    cloner = RepoCloner()
    walker = FileWalker()
    chunker = ASTChunker()
    embedder = Embedder()

    # Clone once
    repo_path = cloner.clone("https://github.com/tiangolo/fastapi")
    files = walker.walk(repo_path)

    # Use small sample for testing
    code_files = [f for f in files if f.file_type == "code"][:15]
    doc_files  = [f for f in files if f.file_type == "doc"][:10]
    chunks = chunker.chunk_many(code_files + doc_files)

    print(f"\nSample chunks:")
    for c in chunks[:5]:
        print(f"  [{c.chunk_type}] {c.name} — {c.file_path} "
              f"(lines {c.start_line}-{c.end_line})")

    # Embed and store
    repo_id = "fastapi_test"
    embedder.setup_collections(repo_id)
    embedder.embed_chunks(chunks, repo_id)
    embedder.get_collection_info(repo_id)

    cloner.cleanup(repo_path)

if __name__ == "__main__":
    test_all()
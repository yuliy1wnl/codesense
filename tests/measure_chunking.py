import time
from backend.ingestion.cloner import RepoCloner
from backend.ingestion.walker import FileWalker
from backend.ingestion.chunker import ASTChunker

def measure_chunking():
    cloner = RepoCloner()
    walker = FileWalker()
    chunker = ASTChunker()

    print("Cloning FastAPI repo...")
    repo_path = cloner.clone("https://github.com/tiangolo/fastapi")
    files = walker.walk(repo_path)
    code_files = [f for f in files if f.file_type == "code"]
    doc_files  = [f for f in files if f.file_type == "doc"]

    print(f"\n── Repo Stats ──────────────────────")
    print(f"  Total files:      {len(files)}")
    print(f"  Code files:       {len(code_files)}")
    print(f"  Doc files:        {len(doc_files)}")
    print(f"  Languages found:  {len(set(f.language for f in code_files))}")
    print(f"  Languages:        {set(f.language for f in code_files)}")

    # AST chunking
    print(f"\nRunning AST chunker...")
    start = time.time()
    ast_chunks = chunker.chunk_many(code_files)
    ast_time = time.time() - start

    ast_sizes = [len(c.content.split("\n")) for c in ast_chunks]
    named = [c for c in ast_chunks
             if c.name != "unknown" and not c.name.startswith("block_")]
    functions = [c for c in ast_chunks if c.chunk_type == "function"]
    classes   = [c for c in ast_chunks if c.chunk_type == "class"]
    blocks    = [c for c in ast_chunks if c.chunk_type == "block"]

    print(f"\n── AST Chunking Results ────────────")
    print(f"  Total chunks:         {len(ast_chunks)}")
    print(f"  Function chunks:      {len(functions)}")
    print(f"  Class chunks:         {len(classes)}")
    print(f"  Fallback blocks:      {len(blocks)}")
    print(f"  Named chunks:         {len(named)} / {len(ast_chunks)}")
    print(f"  Avg chunk size:       {sum(ast_sizes)/len(ast_sizes):.1f} lines")
    print(f"  Min chunk size:       {min(ast_sizes)} lines")
    print(f"  Max chunk size:       {max(ast_sizes)} lines")
    print(f"  Chunking time:        {ast_time:.1f}s")

    # Naive chunking (what most RAG tutorials do)
    print(f"\nRunning naive chunker...")
    start = time.time()
    naive_chunks = []
    for f in code_files:
        lines = f.content.split("\n")
        for i in range(0, len(lines), 40):
            block = "\n".join(lines[i:i+40]).strip()
            if block:
                naive_chunks.append(block)
    naive_time = time.time() - start

    naive_sizes = [len(c.split("\n")) for c in naive_chunks]

    print(f"\n── Naive Chunking Results ──────────")
    print(f"  Total chunks:         {len(naive_chunks)}")
    print(f"  Avg chunk size:       {sum(naive_sizes)/len(naive_sizes):.1f} lines")
    print(f"  Chunking time:        {naive_time:.1f}s")

    # Comparison
    reduction = ((len(naive_chunks) - len(ast_chunks)) / len(naive_chunks)) * 100
    print(f"\n── Comparison ──────────────────────")
    print(f"  AST vs naive chunks:  {len(ast_chunks)} vs {len(naive_chunks)}")
    print(f"  Chunk reduction:      {reduction:.1f}%")
    print(f"  Named/meaningful:     {len(named)/len(ast_chunks)*100:.1f}% of AST chunks")
    print(f"  Named/meaningful:     0% of naive chunks (no names)")

    cloner.cleanup(repo_path)

if __name__ == "__main__":
    measure_chunking()
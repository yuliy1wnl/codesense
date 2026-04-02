import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Citation {
  file: string;
  name: string;
  lines: string;
  type: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  question: string;
}

export interface IngestResponse {
  repo_id: string;
  total_chunks: number;
  code_chunks: number;
  doc_chunks: number;
  message: string;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

// Ingest a GitHub repo
export async function ingestRepo(githubUrl: string): Promise<IngestResponse> {
  const response = await axios.post(`${API_BASE}/ingest`, {
    github_url: githubUrl,
  });
  return response.data;
}

// Ask a question about an ingested repo
export async function queryRepo(
  repoId: string,
  question: string
): Promise<QueryResponse> {
  const response = await axios.post(`${API_BASE}/query`, {
    repo_id: repoId,
    question,
  });
  return response.data;
}

// Stream a response token by token
export async function* queryRepoStream(
  repoId: string,
  question: string
): AsyncGenerator<string> {
  const response = await fetch(`${API_BASE}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_id: repoId, question }),
  });

  if (!response.body) return;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    yield decoder.decode(value);
  }
}

// List all indexed repos
export async function listRepos(): Promise<string[]> {
  const response = await axios.get(`${API_BASE}/repos`);
  return response.data.repos;
}
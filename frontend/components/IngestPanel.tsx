"use client";
import { useState } from "react";
import { GitBranch, Loader2, CheckCircle } from "lucide-react";
import { ingestRepo, IngestResponse } from "@/lib/api";

interface Props {
  onIngested: (repoId: string, stats: IngestResponse) => void;
}

export default function IngestPanel({ onIngested }: Props) {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  const handleIngest = async () => {
    if (!url.trim()) return;
    setLoading(true);
    setError("");
    setStatus("Cloning repository...");

    try {
      setStatus("Chunking and embedding code...");
      const result = await ingestRepo(url.trim());
      setStatus("Done!");
      onIngested(result.repo_id, result);
    } catch (e: any) {
      setError(e?.response?.data?.detail || "Failed to ingest repository");
      setStatus("");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4 p-6 border border-gray-200 
                    rounded-xl bg-white shadow-sm">
      <div className="flex items-center gap-2">
        <GitBranch className="w-5 h-5 text-indigo-500" />
        <h2 className="font-semibold text-gray-800">Index a Repository</h2>
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleIngest()}
          placeholder="https://github.com/owner/repo"
          className="flex-1 px-3 py-2 text-sm border border-gray-200 
                     rounded-lg focus:outline-none focus:ring-2 
                     focus:ring-indigo-300 text-gray-800"
        />
        <button
          onClick={handleIngest}
          disabled={loading || !url.trim()}
          className="px-4 py-2 text-sm font-medium text-white 
                     bg-indigo-500 rounded-lg hover:bg-indigo-600 
                     disabled:opacity-50 disabled:cursor-not-allowed
                     flex items-center gap-2"
        >
          {loading && <Loader2 className="w-4 h-4 animate-spin" />}
          {loading ? "Indexing..." : "Index"}
        </button>
      </div>

      {status && (
        <div className="flex items-center gap-2 text-sm text-indigo-600">
          {loading
            ? <Loader2 className="w-4 h-4 animate-spin" />
            : <CheckCircle className="w-4 h-4" />
          }
          {status}
        </div>
      )}

      {error && (
        <p className="text-sm text-red-500">{error}</p>
      )}
    </div>
  );
}
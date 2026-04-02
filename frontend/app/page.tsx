"use client";
import { useState } from "react";
import { Code2 } from "lucide-react";
import IngestPanel from "@/components/IngestPanel";
import Chat from "@/components/Chat";
import { IngestResponse } from "@/lib/api";

export default function Home() {
  const [repoId, setRepoId] = useState<string | null>(null);
  const [stats, setStats] = useState<IngestResponse | null>(null);

  const handleIngested = (id: string, s: IngestResponse) => {
    setRepoId(id);
    setStats(s);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-100 px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center 
                        justify-between">
          <div className="flex items-center gap-2">
            <Code2 className="w-6 h-6 text-indigo-500" />
            <span className="font-semibold text-gray-800 text-lg">
              CodeSense
            </span>
          </div>
          {stats && (
            <div className="flex gap-4 text-xs text-gray-400">
              <span>{stats.code_chunks} code chunks</span>
              <span>{stats.doc_chunks} doc chunks</span>
              <span className="text-indigo-500 font-medium">
                {stats.repo_id}
              </span>
            </div>
          )}
        </div>
      </header>

      <main className="flex-1 max-w-5xl mx-auto w-full px-6 py-6 
                       flex flex-col gap-6">
        {/* Ingest panel — always visible */}
        <IngestPanel onIngested={handleIngested} />

        {/* Chat — appears after ingestion */}
        {repoId ? (
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden"
                style={{ height: "calc(100vh - 280px)" }}>
              <Chat repoId={repoId} />
            </div>
        ) : (
          <div className="flex-1 flex items-center justify-center 
                          text-gray-300 text-sm">
            Index a repository above to start chatting
          </div>
        )}
      </main>
    </div>
  );
}
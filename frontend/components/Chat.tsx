"use client";
import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Loader2, FileCode, ChevronDown } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { queryRepoStream, Message, Citation } from "@/lib/api";

interface Props {
  repoId: string;
}

function CitationBadge({ citation }: { citation: Citation }) {
  return (
    <div className="flex items-start gap-2 p-2 bg-gray-50
                    rounded-lg border border-gray-100 text-xs">
      <FileCode className="w-3 h-3 mt-0.5 text-indigo-400 flex-shrink-0" />
      <div>
        <p className="font-medium text-gray-700 break-all">{citation.file}</p>
        <p className="text-gray-400">
          {citation.name} · lines {citation.lines}
        </p>
      </div>
    </div>
  );
}

export default function Chat({ repoId }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Show scroll button when user scrolls up
  const handleScroll = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    const distanceFromBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight;
    setShowScrollBtn(distanceFromBottom > 100);
  }, []);

  // Auto scroll to bottom on new messages only if already at bottom
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    const distanceFromBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight;
    if (distanceFromBottom < 100) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    setShowScrollBtn(false);
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const question = input.trim();
    setInput("");
    setLoading(true);

    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", citations: [] },
    ]);

    // Scroll to bottom when new message starts
    setTimeout(() => {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 50);

    try {
      let fullAnswer = "";

      for await (const token of queryRepoStream(repoId, question)) {
        fullAnswer += token;
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: "assistant",
            content: fullAnswer,
            citations: [],
          };
          return updated;
        });
      }

      const citationRes = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo_id: repoId, question }),
      });
      const citationData = await citationRes.json();

      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: fullAnswer,
          citations: citationData.citations || [],
        };
        return updated;
      });
    } catch (e) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: "Sorry, something went wrong. Please try again.",
          citations: [],
        };
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  const suggestions = [
    "Explain the overall architecture",
    "How does routing work?",
    "Where is authentication handled?",
    "How do I run this project?",
  ];

  return (
    <div className="flex flex-col h-full">

      {/* Scrollable messages area */}
      <div
        className="flex-1 overflow-y-auto p-4 space-y-4 relative"
        ref={scrollContainerRef}
        onScroll={handleScroll}
      >
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center
                          h-full gap-4 text-center">
            <p className="text-gray-400 text-sm">
              Ask anything about{" "}
              <span className="font-medium text-indigo-500">{repoId}</span>
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {suggestions.map((s) => (
                <button
                  key={s}
                  onClick={() => setInput(s)}
                  className="px-3 py-1.5 text-xs border border-gray-200
                             rounded-full text-gray-500 hover:border-indigo-300
                             hover:text-indigo-500 transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${
              msg.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-[80%] ${
                msg.role === "user"
                  ? "bg-indigo-500 text-white rounded-2xl rounded-tr-sm px-4 py-2"
                  : "w-full"
              }`}
            >
              {msg.role === "user" ? (
                <p className="text-sm">{msg.content}</p>
              ) : (
                <div className="space-y-3">
                  <div className="prose prose-sm max-w-none text-gray-700">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                    {loading && i === messages.length - 1 && (
                      <span
                        className="inline-block w-1.5 h-4 bg-indigo-400
                                   animate-pulse ml-0.5 align-middle"
                      />
                    )}
                  </div>

                  {msg.citations && msg.citations.length > 0 && (
                    <div className="space-y-1.5">
                      <p className="text-xs font-medium text-gray-400
                                    uppercase tracking-wide">
                        Sources
                      </p>
                      {msg.citations.map((c, j) => (
                        <CitationBadge key={j} citation={c} />
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      {/* Scroll to bottom button — floats over messages, doesn't affect layout */}
      {showScrollBtn && (
        <div className="absolute bottom-20 left-1/2 -translate-x-1/2 z-10">
          <button
            onClick={scrollToBottom}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs
                      bg-white border border-gray-200 rounded-full
                      text-gray-500 hover:border-indigo-300
                      hover:text-indigo-500 shadow-md transition-all
                      whitespace-nowrap"
          >
            <ChevronDown className="w-3 h-3" />
            Scroll to bottom
          </button>
        </div>
      )}

      {/* Sticky input — always at bottom */}
      <div className="border-t-2 border-gray-200 p-4 bg-white shadow-[0_-2px_8px_rgba(0,0,0,0.04)]">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Ask about the codebase..."
            disabled={loading}
            className="flex-1 px-4 py-2.5 text-sm border border-gray-200
                       rounded-xl focus:outline-none focus:ring-2
                       focus:ring-indigo-300 disabled:opacity-50 text-gray-800"
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="p-2.5 bg-indigo-500 text-white rounded-xl
                       hover:bg-indigo-600 disabled:opacity-50
                       disabled:cursor-not-allowed"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
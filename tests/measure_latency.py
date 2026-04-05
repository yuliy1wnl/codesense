import time
import requests
import statistics

def measure_latency():
    questions = [
        "How does routing work?",
        "Where is authentication handled?",
        "What does APIRouter do?",
        "How do I run this project?",
        "How does error handling work?"
    ]

    print("Measuring query latency...")
    print("(each question will take 20-60 seconds)\n")

    latencies = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/5] {q}")
        start = time.time()
        response = requests.post(
            "http://localhost:8000/query",
            json={"repo_id": "fastapi", "question": q}
        )
        elapsed = time.time() - start
        latencies.append(elapsed)
        print(f"       {elapsed:.1f}s\n")

    print(f"── Latency Results ─────────────────")
    print(f"  Average:   {statistics.mean(latencies):.1f}s")
    print(f"  Median:    {statistics.median(latencies):.1f}s")
    print(f"  Fastest:   {min(latencies):.1f}s")
    print(f"  Slowest:   {max(latencies):.1f}s")
    print(f"  Std dev:   {statistics.stdev(latencies):.1f}s")

if __name__ == "__main__":
    measure_latency()
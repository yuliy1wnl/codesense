import requests

def test_evaluation():
    payload = {
        "repo_id": "fastapi",
        "questions": [
            "How does routing work in this project?",
            "How do I run this project?",
            "What does the APIRouter class do?"
        ],
        "ground_truths": [
            "FastAPI routing uses the api_route decorator and APIRoute class to match requests to handlers",
            "Install dependencies with pip and run with uvicorn",
            "APIRouter allows grouping of related routes and is used to organize endpoints"
        ]
    }

    print("Running evaluation — this will take 2-3 minutes...")
    response = requests.post(
        "http://localhost:8000/evaluate",
        json=payload
    )

    # Print raw response first so we can see any errors
    print(f"Status code: {response.status_code}")
    print(f"Raw response: {response.text[:500]}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nScores:")
        for metric, score in result["scores"].items():
            if metric != "eval_time_seconds":
                print(f"  {metric:<25} {score}")
    else:
        print("Request failed — check uvicorn terminal for traceback")

if __name__ == "__main__":
    test_evaluation()
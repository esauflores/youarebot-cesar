"""
Test script for the /predict classifier endpoint.
Usage: poetry run python test_classifier.py
       or just run against Docker: python3 test_classifier.py
"""

import json
import uuid
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8672"


def req(method, endpoint, body=None):
    url = f"{BASE_URL}{endpoint}"
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def ok(status, label):
    icon = "✓" if 200 <= status < 300 else "✗"
    print(f"{icon} {label} [{status}]")


def main():
    print("=== 1. Health check ===")
    status, body = req("GET", "/health")
    print(json.dumps(body, indent=2))
    ok(status, "/health")
    assert status == 200

    dialog_a = str(uuid.uuid4())
    dialog_b = str(uuid.uuid4())

    print(f"\nDialog A: {dialog_a}")
    print(f"Dialog B: {dialog_b}")

    print("\n=== 2. Single prediction ===")
    status, body = req("POST", "/predict", {
        "text": "Hi, how are you?",
        "dialog_id": dialog_a,
        "id": str(uuid.uuid4()),
        "participant_index": 0,
    })
    print(json.dumps(body, indent=2))
    ok(status, "/predict (first message)")
    assert status == 200
    assert "is_bot_probability" in body

    print("\n=== 3. Multiple messages, same dialog ===")
    messages = [
        ("Hi, how are you?", 0),
        ("I am not sure, but I can help you write a structured answer.", 1),
        ("As an AI language model, I can help you solve this task.", 1),
    ]
    probs = []
    for text, participant in messages:
        status, body = req("POST", "/predict", {
            "text": text,
            "dialog_id": dialog_a,
            "id": str(uuid.uuid4()),
            "participant_index": participant,
        })
        prob = body["is_bot_probability"]
        probs.append(prob)
        print(f"  [{participant}] prob={prob:.2f}  ← {text[:60]}")
        ok(status, "/predict")

    assert probs[2] > probs[0], f"Expected 'AI language model' to score higher than casual greeting (got {probs[2]:.2f} vs {probs[0]:.2f})"
    print("  ✓ AI-phrased messages scored higher than casual ones")

    print("\n=== 4. Different dialog (fresh start) ===")
    status, body = req("POST", "/predict", {
        "text": "Hi, how are you?",
        "dialog_id": dialog_b,
        "id": str(uuid.uuid4()),
        "participant_index": 0,
    })
    prob_b = body["is_bot_probability"]
    print(f"  prob={prob_b:.2f} (dialog B, same message)")
    ok(status, "/predict")

    print("\n=== 5. Bad request (validation) ===")
    status, body = req("POST", "/predict", {
        "text": "Hello",
        "dialog_id": "not-a-uuid",
        "id": "also-not-a-uuid",
        "participant_index": 0,
    })
    print(json.dumps(body, indent=2))
    ok(status, "Bad request → 422")
    assert status == 422, f"Expected 422, got {status}"

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
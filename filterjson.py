import json
import sys

def filter_json(first, second, fields=("inputs", "targets", "metric", "domain", "task")):
    # Build a set of tuples for all specified fields from the second JSON
    reference_pairs = {
        tuple(entry[field] for field in fields)
        for entry in second
        if all(field in entry for field in fields)
    }

    # Keep entries where ALL field values match exactly within the same entry in second JSON
    filtered = [
        entry for entry in first
        if tuple(entry.get(field) for field in fields) in reference_pairs
    ]

    return filtered


def main():
    # ─── Option 1: Pass file paths as arguments ───────────────────────────────
    # Usage: python filter_json.py first.json second.json [output.json]
    if len(sys.argv) >= 3:
        with open(sys.argv[1], "r") as f:
            first = json.load(f)
        with open(sys.argv[2], "r") as f:
            second = json.load(f)

        result = filter_json(first, second)

        if len(sys.argv) >= 4:
            with open(sys.argv[3], "w") as f:
                json.dump(result, f, indent=2)
            print(f"Filtered result saved to: {sys.argv[3]}")
        else:
            print(json.dumps(result, indent=2))

    # ─── Option 2: Hardcode your JSONs directly below ─────────────────────────
    else:
        first = [
            {"id": 1, "inputs": "apple",  "targets": "red",    "metric": "accuracy", "domain": "food", "task": "classify"},
            {"id": 2, "inputs": "banana", "targets": "yellow", "metric": "f1",        "domain": "food", "task": "classify"},
            {"id": 3, "inputs": "cherry", "targets": "red",    "metric": "accuracy", "domain": "food", "task": "classify"},
            {"id": 4, "inputs": "grape",  "targets": "purple", "metric": "accuracy", "domain": "food", "task": "detect"},
        ]

        # Entry 1: all 5 fields match exactly        → KEPT
        # Entry 2: inputs/targets differ             → REMOVED
        # Entry 3: targets match but metric differs  → REMOVED (second has metric="f1")
        # Entry 4: task differs                      → REMOVED (second has task="classify")
        second = [
            {"id": 1, "inputs": "apple",  "targets": "red",    "metric": "accuracy", "domain": "food", "task": "classify"},
            {"id": 3, "inputs": "cherry", "targets": "red",    "metric": "f1",       "domain": "food", "task": "classify"},
            {"id": 4, "inputs": "grape",  "targets": "purple", "metric": "accuracy", "domain": "food", "task": "classify"},
        ]

        result = filter_json(first, second)

        print("=== First JSON ===")
        print(json.dumps(first, indent=2))
        print("\n=== Second JSON (reference) ===")
        print(json.dumps(second, indent=2))
        print("\n=== Filtered Result ===")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
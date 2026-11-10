#!/usr/bin/env python3
"""
Summarize the 15 sampled test solutions per problem
===================================================
Recomputes, from results/group_{high,medium,low}.json only (no models needed):

- Pass@k on the 1,319 GSM8K test problems for k = 1..15, overall and per
  consensus group. Pass@k counts a problem as solved if any of its first k
  samples (in generation order) has the correct final answer.
- Majority-vote accuracy and errors per consensus group (same rule as
  method_majority_vote in src/evaluation/evaluate_selectors.py), and the
  share of all majority-vote errors that falls in each group.

The majority-vote accuracies are checked against results/ranker_eval_results.json.

Output: ./results/test_sample_stats.json

Usage (from the repository root):
    python src/analysis/summarize_test_samples.py
"""

import json
from collections import Counter

GROUPS = ["high", "medium", "low"]
GROUP_PATH = "./results/group_{}.json"
EVAL_RESULTS_PATH = "./results/ranker_eval_results.json"
OUTPUT_PATH = "./results/test_sample_stats.json"
K_MAX = 15


def majority_vote_correct(item):
    answers = [s["predicted_answer"] for s in item["solutions"] if s["predicted_answer"] is not None]
    if not answers:
        return False
    predicted = Counter(answers).most_common(1)[0][0]
    return predicted == item["ground_truth"]


def pass_at_k(items, k):
    solved = sum(1 for it in items if any(s["is_correct"] for s in it["solutions"][:k]))
    return solved


def main():
    groups = {}
    for g in GROUPS:
        with open(GROUP_PATH.format(g)) as f:
            groups[g] = json.load(f)
    all_items = [it for g in GROUPS for it in groups[g]]
    n_total = len(all_items)
    assert all(len(it["solutions"]) == K_MAX for it in all_items), "expected 15 samples per problem"

    stats = {
        "description": (
            "Statistics recomputed from results/group_*.json "
            "(1,319 GSM8K test problems x 15 sampled solutions) by "
            "src/analysis/summarize_test_samples.py. Percentages are 0-100."
        ),
        "n_problems": {"overall": n_total, **{g: len(groups[g]) for g in GROUPS}},
        "pass_at_k_percent": {},
        "pass_at_k_count": {},
        "majority_vote": {},
    }

    scopes = {"overall": all_items, **groups}
    for scope, items in scopes.items():
        counts = {str(k): pass_at_k(items, k) for k in range(1, K_MAX + 1)}
        stats["pass_at_k_count"][scope] = counts
        stats["pass_at_k_percent"][scope] = {k: round(100 * c / len(items), 2) for k, c in counts.items()}

    errors_total = 0
    for scope, items in scopes.items():
        correct = sum(majority_vote_correct(it) for it in items)
        stats["majority_vote"][scope] = {
            "correct": correct,
            "errors": len(items) - correct,
            "accuracy_percent": round(100 * correct / len(items), 2),
        }
    errors_total = stats["majority_vote"]["overall"]["errors"]
    for g in GROUPS:
        e = stats["majority_vote"][g]["errors"]
        stats["majority_vote"][g]["share_of_all_errors_percent"] = round(100 * e / errors_total, 2)
        stats["majority_vote"][g]["share_of_problems_percent"] = round(100 * len(groups[g]) / n_total, 2)

    # Cross-check against the stored evaluation results.
    with open(EVAL_RESULTS_PATH) as f:
        stored = json.load(f)["accuracy"]["mv"]
    for scope in ["overall"] + GROUPS:
        ours = stats["majority_vote"][scope]["correct"] / len(scopes[scope])
        assert abs(ours - stored[scope]) < 1e-9, f"majority vote mismatch in {scope}: {ours} vs {stored[scope]}"

    with open(OUTPUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")

    print(f"Saved {OUTPUT_PATH}")
    print("Pass@k (overall):", {k: stats["pass_at_k_percent"]["overall"][str(k)] for k in [1, 3, 5, 7, 10, 15]})
    for g in GROUPS:
        mv = stats["majority_vote"][g]
        print(f"{g:>6}: MV {mv['accuracy_percent']}%, errors {mv['errors']} "
              f"({mv['share_of_all_errors_percent']}% of all errors), "
              f"Pass@15 {stats['pass_at_k_percent'][g]['15']}%")


if __name__ == "__main__":
    main()

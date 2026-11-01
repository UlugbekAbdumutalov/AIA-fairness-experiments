import csv
import json
import math
from pathlib import Path
from bisect import bisect_left, bisect_right
from statistics import mean, median

ROOT = Path.home() / "pan18_project"
OLD = ROOT / "phase1_recovered/v5_Pythia-6.9B_author_scores_PROVISIONAL.csv"
NEW = ROOT / "phase1_overnight/pythia_3000_single_scores.csv"
OUTPUT = ROOT / "phase1_overnight/pythia_full_comparison.json"

def load(path, score_column):
    data = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = row["author_id"]
            if aid in data:
                raise ValueError(f"Duplicate author: {aid}")
            gender = row["true_gender"]
            score = float(row[score_column])
            if gender not in ("male", "female") or not math.isfinite(score):
                raise ValueError(f"Invalid record: {aid}")
            data[aid] = (gender, score)
    return data

def auc(male, female):
    female = sorted(female)
    wins = 0
    for score in male:
        lo = bisect_left(female, score)
        hi = bisect_right(female, score)
        wins += lo + (hi - lo) / 2
    return wins / (len(male) * len(female))

def metrics(data):
    male = [score for gender, score in data.values() if gender == "male"]
    female = [score for gender, score in data.values() if gender == "female"]

    male_acc = sum(x > 0 for x in male) / len(male)
    female_acc = sum(x <= 0 for x in female) / len(female)

    return {
        "male_authors": len(male),
        "female_authors": len(female),
        "auc": auc(male, female),
        "male_accuracy": male_acc,
        "female_accuracy": female_acc,
        "signed_gap": male_acc - female_acc,
        "average_accuracy": (male_acc + female_acc) / 2
    }

old = load(OLD, "mean_margin")
new = load(NEW, "margin")

assert len(old) == len(new) == 3000
assert set(old) == set(new), "Author IDs do not match"

for aid in old:
    assert old[aid][0] == new[aid][0], f"Gender mismatch: {aid}"

old_metrics = metrics(old)
new_metrics = metrics(new)

diffs = [abs(old[aid][1] - new[aid][1]) for aid in old]
flips = sum(
    (old[aid][1] > 0) != (new[aid][1] > 0)
    for aid in old
)

print("=" * 65)
print("PYTHIA — FULL 3,000-AUTHOR COMPARISON")
print("=" * 65)

for label, m in (("OLD (averaged batch scores)", old_metrics),
                 ("NEW (single-author scores)", new_metrics)):
    print(f"\n{label}")
    print(f"  AUC:              {m['auc']:.6f}")
    print(f"  Male accuracy:    {m['male_accuracy']:.6f}")
    print(f"  Female accuracy:  {m['female_accuracy']:.6f}")
    print(f"  Signed gap:       {m['signed_gap']:+.6f}")
    print(f"  Average accuracy: {m['average_accuracy']:.6f}")

print("\nCHANGES")
print(f"  AUC change:           {new_metrics['auc'] - old_metrics['auc']:+.6f}")
print(f"  Accuracy gap change:  {new_metrics['signed_gap'] - old_metrics['signed_gap']:+.6f}")
print(f"  Mean score change:    {mean(diffs):.6f}")
print(f"  Median score change:  {median(diffs):.6f}")
print(f"  Maximum score change: {max(diffs):.6f}")
print(f"  Prediction changes:   {flips}/3000 ({100*flips/3000:.2f}%)")

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump({
        "old": old_metrics,
        "new": new_metrics,
        "auc_change": new_metrics["auc"] - old_metrics["auc"],
        "gap_change": new_metrics["signed_gap"] - old_metrics["signed_gap"],
        "mean_absolute_score_change": mean(diffs),
        "median_absolute_score_change": median(diffs),
        "maximum_absolute_score_change": max(diffs),
        "prediction_changes": flips,
    }, f, indent=2)

print("\nSaved:", OUTPUT)
print("Original experiment files were not changed.")

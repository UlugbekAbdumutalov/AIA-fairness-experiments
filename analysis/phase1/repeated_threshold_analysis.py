import csv
import random
from pathlib import Path
from statistics import mean, median

ROOT = Path.home() / "pan18_project"
SOURCE = ROOT / "phase1_recovered"
OUT = ROOT / "phase1_threshold_results"
OUT.mkdir(exist_ok=True)

MODELS = {
    "GPT-2": "v6_GPT-2_author_scores_PROVISIONAL.csv",
    "GPT-2-Medium": "v6_GPT-2-Medium_author_scores_PROVISIONAL.csv",
    "BERT": "BERT_author_scores_PROVISIONAL.csv",
    "RoBERTa": "RoBERTa_author_scores_PROVISIONAL.csv",
    "BART": "BART_author_scores_PROVISIONAL.csv",
}

N_SPLITS = 200

def read_authors(filename):
    groups = {"male": [], "female": []}
    seen = set()

    with (SOURCE / filename).open(newline="") as f:
        for row in csv.DictReader(f):
            author = row["author_id"]
            gender = row["true_gender"]
            score = float(row["mean_margin"])

            if author in seen:
                raise ValueError("Duplicate author: " + author)
            seen.add(author)
            groups[gender].append((author, score))

    return groups

def evaluate(male, female, cutoff):
    m = sum(x > cutoff for x in male) / len(male)
    f = sum(x <= cutoff for x in female) / len(female)
    return m - f, (m + f) / 2

def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    weight = index - lo
    return values[lo] * (1 - weight) + values[hi] * weight

all_rows = []

for model, filename in MODELS.items():
    groups = read_authors(filename)
    model_rows = []

    for split in range(N_SPLITS):
        rng = random.Random(2026 + split)
        validation = {}
        test = {}

        for gender in ("male", "female"):
            authors = sorted(groups[gender])
            rng.shuffle(authors)
            n = len(authors) // 2

            validation[gender] = [x[1] for x in authors[:n]]
            test[gender] = [x[1] for x in authors[n:]]

        val_scores = validation["male"] + validation["female"]
        cutoff = median(val_scores)

        male = test["male"]
        female = test["female"]

        old_gap, old_acc = evaluate(male, female, 0.0)
        new_gap, new_acc = evaluate(male, female, cutoff)

        row = {
            "model": model,
            "split": split,
            "cutoff": cutoff,
            "old_gap": old_gap,
            "new_gap": new_gap,
            "old_accuracy": old_acc,
            "new_accuracy": new_acc,
            "accuracy_change": new_acc - old_acc,
            "absolute_gap_reduction": abs(old_gap) - abs(new_gap),
        }

        model_rows.append(row)
        all_rows.append(row)

    reductions = [r["absolute_gap_reduction"] for r in model_rows]
    changes = [r["accuracy_change"] for r in model_rows]
    new_gaps = [r["new_gap"] for r in model_rows]

    print("\n" + "=" * 60)
    print("MODEL:", model)
    print("Unique authors:", sum(len(x) for x in groups.values()))
    print("Number of splits:", N_SPLITS)

    print("Mean original signed gap:",
          f"{mean(r['old_gap'] for r in model_rows):+.4f}")
    print("Mean adjusted signed gap:",
          f"{mean(new_gaps):+.4f}")
    print("Median absolute adjusted gap:",
          f"{median(abs(x) for x in new_gaps):.4f}")

    print("Mean absolute gap reduction:",
          f"{mean(reductions):+.4f}")
    print("Gap reduction range (5th-95th percentile):",
          f"{percentile(reductions, .05):+.4f}",
          "to",
          f"{percentile(reductions, .95):+.4f}")

    print("Mean accuracy change:",
          f"{mean(changes):+.4f}")
    print("Accuracy change range (5th-95th percentile):",
          f"{percentile(changes, .05):+.4f}",
          "to",
          f"{percentile(changes, .95):+.4f}")

    print("Splits with smaller absolute gap:",
          f"{sum(x > 0 for x in reductions)}/{N_SPLITS}")

path = OUT / "repeated_threshold_200.csv"

with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    writer.writeheader()
    writer.writerows(all_rows)

print("\n" + "=" * 60)
print("FINISHED")
print("Saved:", path)
print("Note: percentile ranges describe repeated splits.")
print("They are NOT independent-sample confidence intervals.")

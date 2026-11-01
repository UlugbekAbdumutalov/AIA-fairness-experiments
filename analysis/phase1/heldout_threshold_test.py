import csv
import json
import random
from pathlib import Path
from statistics import median
from bisect import bisect_left, bisect_right

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

SEED = 2026


def load_data(filename):
    groups = {"male": [], "female": []}
    seen = set()

    with (SOURCE / filename).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            author_id = row["author_id"]
            gender = row["true_gender"]
            score = float(row["mean_margin"])

            if author_id in seen:
                raise ValueError(f"Duplicate author: {author_id}")
            if gender not in groups:
                raise ValueError(f"Unknown gender: {gender}")

            seen.add(author_id)
            groups[gender].append((author_id, score))

    return groups


def split_data(groups):
    rng = random.Random(SEED)
    validation = {}
    test = {}

    for gender in ("male", "female"):
        authors = sorted(groups[gender])
        rng.shuffle(authors)

        midpoint = len(authors) // 2
        validation[gender] = authors[:midpoint]
        test[gender] = authors[midpoint:]

        val_ids = {x[0] for x in validation[gender]}
        test_ids = {x[0] for x in test[gender]}
        assert val_ids.isdisjoint(test_ids)

    return validation, test


def get_scores(data, gender):
    return [score for _, score in data[gender]]


def calculate_auc(male, female):
    female_sorted = sorted(female)
    wins = 0.0

    for score in male:
        below = bisect_left(female_sorted, score)
        equal = bisect_right(female_sorted, score) - below
        wins += below + 0.5 * equal

    return wins / (len(male) * len(female))


def evaluate(data, cutoff):
    male = get_scores(data, "male")
    female = get_scores(data, "female")

    male_acc = sum(x > cutoff for x in male) / len(male)
    female_acc = sum(x <= cutoff for x in female) / len(female)

    return {
        "male_accuracy": male_acc,
        "female_accuracy": female_acc,
        "gap": male_acc - female_acc,
        "balanced_accuracy": (male_acc + female_acc) / 2,
        "auc": calculate_auc(male, female),
        "predicted_male_rate": (
            sum(x > cutoff for x in male + female)
            / (len(male) + len(female))
        ),
    }


results = []

print("=" * 70)
print("PHASE 1 — AUTHOR-DISJOINT THRESHOLD TEST")
print("CPU only | No model inference | No original files changed")
print("=" * 70)

for name, filename in MODELS.items():
    groups = load_data(filename)
    validation, test = split_data(groups)

    val_scores = (
        get_scores(validation, "male")
        + get_scores(validation, "female")
    )

    validation_cutoff = median(val_scores)

    default_result = evaluate(test, 0.0)
    adjusted_result = evaluate(test, validation_cutoff)

    print(f"\nMODEL: {name}")
    print(
        f"Validation: {len(validation['male'])} M + "
        f"{len(validation['female'])} F"
    )
    print(
        f"Test:       {len(test['male'])} M + "
        f"{len(test['female'])} F"
    )
    print(f"Validation median cut-off: {validation_cutoff:+.6f}")
    print(f"Test AUC: {default_result['auc']:.4f}")

    for label, result in (
        ("Zero cut-off", default_result),
        ("Validation median", adjusted_result),
    ):
        print(f"  {label}:")
        print(f"    Male accuracy:   {result['male_accuracy']:.4f}")
        print(f"    Female accuracy: {result['female_accuracy']:.4f}")
        print(f"    Signed gap:      {result['gap']:+.4f}")
        print(f"    Average accuracy:{result['balanced_accuracy']:.4f}")
        print(f"    Predicted male:  {result['predicted_male_rate']:.4f}")

        results.append({
            "model": name,
            "method": label,
            "validation_male_n": len(validation["male"]),
            "validation_female_n": len(validation["female"]),
            "test_male_n": len(test["male"]),
            "test_female_n": len(test["female"]),
            "cutoff": 0.0 if label == "Zero cut-off"
                      else validation_cutoff,
            **result,
        })

    print(
        "  Change in absolute gap: "
        f"{abs(adjusted_result['gap']) - abs(default_result['gap']):+.4f}"
    )
    print(
        "  Change in average accuracy: "
        f"{adjusted_result['balanced_accuracy'] - default_result['balanced_accuracy']:+.4f}"
    )

csv_path = OUT / "threshold_test_results.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

json_path = OUT / "threshold_test_results.json"
with json_path.open("w", encoding="utf-8") as f:
    json.dump(
        {"seed": SEED, "results": results},
        f,
        indent=2,
    )

print("\n" + "=" * 70)
print("FINISHED")
print("CSV:", csv_path)
print("JSON:", json_path)
print("These are single-split results, not confidence intervals.")
print("Pythia excluded pending scoring stability investigation.")

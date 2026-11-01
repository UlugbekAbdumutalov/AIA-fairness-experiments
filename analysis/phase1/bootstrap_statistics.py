import csv
import json
import numpy as np
from pathlib import Path

ROOT = Path.home() / "pan18_project"
SOURCE = ROOT / "phase1_recovered"
OUT = ROOT / "phase1_threshold_results"
OUT.mkdir(exist_ok=True)

MODELS = {
    "GPT-2": "v6_GPT-2_author_scores_PROVISIONAL.csv",
    "GPT-2-Medium": "v6_GPT-2-Medium_author_scores_PROVISIONAL.csv",
    "Pythia-6.9B": "v5_Pythia-6.9B_author_scores_PROVISIONAL.csv",
    "BERT": "BERT_author_scores_PROVISIONAL.csv",
    "RoBERTa": "RoBERTa_author_scores_PROVISIONAL.csv",
    "BART": "BART_author_scores_PROVISIONAL.csv",
}

N_BOOT = 2000
SEED = 2026

def load_data(filename):
    groups = {"male": [], "female": []}
    ids = set()

    with (SOURCE / filename).open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            aid = row["author_id"]
            gender = row["true_gender"]
            score = float(row["mean_margin"])

            if aid in ids:
                raise ValueError(
                    f"Duplicate author ID: {aid}"
                )
            if gender not in groups:
                raise ValueError(
                    f"Unexpected gender: {gender}"
                )
            if not np.isfinite(score):
                raise ValueError(
                    f"Invalid score for author {aid}"
                )

            ids.add(aid)
            groups[gender].append(score)

    male = np.array(groups["male"], dtype=float)
    female = np.array(groups["female"], dtype=float)

    if not len(male) or not len(female):
        raise ValueError("One gender group is empty")

    return male, female

def calculate_auc(male, female):
    # Each male score is compared to every female score.
    # Equal scores count as half a win.
    sorted_f = np.sort(female)

    below = np.searchsorted(
        sorted_f, male, side="left"
    )
    equal_end = np.searchsorted(
        sorted_f, male, side="right"
    )

    wins = below + 0.5 * (equal_end - below)

    return float(
        np.sum(wins) / (len(male) * len(female))
    )

def calculate_metrics(male, female):
    male_acc = float(np.mean(male > 0))
    female_acc = float(np.mean(female <= 0))

    return {
        "auc": calculate_auc(male, female),
        "male_accuracy": male_acc,
        "female_accuracy": female_acc,
        "signed_gap": male_acc - female_acc,
        "absolute_gap": abs(male_acc - female_acc),
        "balanced_accuracy": (
            male_acc + female_acc
        ) / 2,
    }

def confidence_interval(values):
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )

all_results = []

print("=" * 70)
print("PHASE 1 — AUTHOR-LEVEL BOOTSTRAP")
print(f"Bootstrap samples: {N_BOOT}")
print("Random seed:", SEED)
print("No GPU or model inference needed")
print("=" * 70)

for model_index, (name, filename) in enumerate(
    MODELS.items()
):
    male, female = load_data(filename)
    original = calculate_metrics(male, female)

    # Each model gets its own reproducible random stream.
    rng = np.random.default_rng(SEED + model_index)

    distributions = {
        key: []
        for key in original
    }

    for i in range(N_BOOT):
        sample_m = male[
            rng.integers(
                0, len(male), size=len(male)
            )
        ]

        sample_f = female[
            rng.integers(
                0, len(female), size=len(female)
            )
        ]

        metrics = calculate_metrics(
            sample_m, sample_f
        )

        for key, value in metrics.items():
            distributions[key].append(value)

    row = {
        "model": name,
        "male_authors": len(male),
        "female_authors": len(female),
    }

    print("\n" + "=" * 70)
    print("MODEL:", name)
    print(
        "Unique authors:",
        len(male) + len(female)
    )

    for metric, estimate in original.items():
        lo, hi = confidence_interval(
            distributions[metric]
        )

        row[metric] = estimate
        row[f"{metric}_ci_lower"] = lo
        row[f"{metric}_ci_upper"] = hi

        print(
            f"{metric:22s} "
            f"{estimate:+.4f} "
            f"95% CI [{lo:+.4f}, {hi:+.4f}]"
        )

    all_results.append(row)

csv_path = OUT / "author_bootstrap_95ci.csv"

with csv_path.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=all_results[0].keys()
    )
    writer.writeheader()
    writer.writerows(all_results)

json_path = OUT / "author_bootstrap_95ci.json"

with json_path.open(
    "w", encoding="utf-8"
) as f:
    json.dump(
        {
            "method": "stratified author-level percentile bootstrap",
            "bootstrap_samples": N_BOOT,
            "seed": SEED,
            "cutoff": 0.0,
            "results": all_results,
            "limitations": [
                "Scores must be correctly matched to authors.",
                "Pythia scores are repeated-measurement means.",
                "Batch-related score variation is not included.",
                "These intervals do not measure model or dataset shift.",
                "Original generated-answer results are separate."
            ],
        },
        f,
        indent=2,
    )

print("\n" + "=" * 70)
print("BOOTSTRAP FINISHED")
print("CSV:", csv_path)
print("JSON:", json_path)
print("Original experiment files were not changed.")

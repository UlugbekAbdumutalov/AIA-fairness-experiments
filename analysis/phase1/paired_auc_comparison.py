import csv
import json
import numpy as np
from pathlib import Path

ROOT = Path.home() / "pan18_project"
SOURCE = ROOT / "phase1_recovered"
OUT = ROOT / "phase1_threshold_results"
OUT.mkdir(exist_ok=True)

N_BOOT = 2000
SEED = 2026

FILES = {
    "RoBERTa": SOURCE / "RoBERTa_author_scores_PROVISIONAL.csv",
    "BART": SOURCE / "BART_author_scores_PROVISIONAL.csv",
}

def load_model(path):
    result = {}

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            author_id = row["author_id"]
            gender = row["true_gender"]
            score = float(row["mean_margin"])

            if author_id in result:
                raise ValueError(f"Duplicate author: {author_id}")

            if gender not in ("male", "female"):
                raise ValueError(f"Unexpected gender: {gender}")

            if not np.isfinite(score):
                raise ValueError(f"Invalid score: {author_id}")

            result[author_id] = (gender, score)

    return result

def auc(male, female):
    female_sorted = np.sort(female)

    below = np.searchsorted(female_sorted, male, side="left")
    equal_end = np.searchsorted(female_sorted, male, side="right")

    wins = below + 0.5 * (equal_end - below)
    return float(np.sum(wins) / (len(male) * len(female)))

data = {name: load_model(path) for name, path in FILES.items()}

roberta = data["RoBERTa"]
bart = data["BART"]

if set(roberta) != set(bart):
    raise ValueError("Author IDs do not match between models!")

ids = sorted(roberta)

for author_id in ids:
    if roberta[author_id][0] != bart[author_id][0]:
        raise ValueError(f"Gender mismatch: {author_id}")

male_ids = [i for i in ids if roberta[i][0] == "male"]
female_ids = [i for i in ids if roberta[i][0] == "female"]

print("=" * 65)
print("PAIRED ROBERTA–BART BOOTSTRAP")
print("Matched male authors:", len(male_ids))
print("Matched female authors:", len(female_ids))
print("All author IDs and labels match.")
print("=" * 65)

if len(male_ids) != 700 or len(female_ids) != 700:
    raise ValueError("Unexpected sample size.")

def get_scores(model, selected_ids):
    return np.array(
        [model[i][1] for i in selected_ids],
        dtype=float
    )

rm = get_scores(roberta, male_ids)
rf = get_scores(roberta, female_ids)
bm = get_scores(bart, male_ids)
bf = get_scores(bart, female_ids)

auc_r = auc(rm, rf)
auc_b = auc(bm, bf)
observed_difference = auc_r - auc_b

print(f"\nRoBERTa AUC: {auc_r:.6f}")
print(f"BART AUC:    {auc_b:.6f}")
print(f"Difference:  {observed_difference:+.6f}")

rng = np.random.default_rng(SEED)

differences = []
roberta_aucs = []
bart_aucs = []

for _ in range(N_BOOT):
    # Sample the same author indices for both models.
    mi = rng.integers(0, len(male_ids), len(male_ids))
    fi = rng.integers(0, len(female_ids), len(female_ids))

    r_auc = auc(rm[mi], rf[fi])
    b_auc = auc(bm[mi], bf[fi])

    roberta_aucs.append(r_auc)
    bart_aucs.append(b_auc)
    differences.append(r_auc - b_auc)

lower, upper = np.percentile(differences, [2.5, 97.5])

print("\n" + "=" * 65)
print("FINAL PAIRED COMPARISON")
print("=" * 65)
print(f"RoBERTa AUC:          {auc_r:.6f}")
print(f"BART AUC:             {auc_b:.6f}")
print(f"AUC difference:       {observed_difference:+.6f}")
print(f"95% CI for difference:[{lower:+.6f}, {upper:+.6f}]")

if lower <= 0 <= upper:
    print("INTERPRETATION: The interval includes zero.")
    print("We cannot establish an AUC difference from these data.")
else:
    print("INTERPRETATION: The interval excludes zero.")
    print("The observed AUC difference is distinguishable from zero")
    print("under this author-level bootstrap procedure.")

result = {
    "method": "paired stratified author-level percentile bootstrap",
    "models": ["RoBERTa", "BART"],
    "matched_male_authors": len(male_ids),
    "matched_female_authors": len(female_ids),
    "bootstrap_samples": N_BOOT,
    "seed": SEED,
    "roberta_auc": auc_r,
    "bart_auc": auc_b,
    "auc_difference": observed_difference,
    "ci_lower": float(lower),
    "ci_upper": float(upper),
    "limitations": [
        "Only PAN 2018 English authors are evaluated.",
        "Recovered author-score alignment depends on original ordering.",
        "The interval measures author-sampling uncertainty.",
        "It does not establish architecture effects."
    ],
}

path = OUT / "roberta_bart_paired_auc.json"

with path.open("w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

csv_path = OUT / "roberta_bart_paired_auc.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "roberta_auc",
            "bart_auc",
            "auc_difference",
            "ci_lower",
            "ci_upper",
            "matched_male_authors",
            "matched_female_authors",
            "bootstrap_samples",
        ]
    )
    writer.writeheader()
    writer.writerow({
        key: result[key]
        for key in writer.fieldnames
    })

print("\nResults saved:")
print(path)
print(csv_path)
print("\nOriginal files were not changed.")

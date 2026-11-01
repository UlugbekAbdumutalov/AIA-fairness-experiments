import json
import csv
import random
from pathlib import Path

ROOT = Path.home() / "pan18_project"
DATA = Path.home() / "pan18/pan18-author-profiling-training-dataset-2018-02-27/en"
OUT = ROOT / "phase1_recovered"
OUT.mkdir(exist_ok=True)

groups = {"male": [], "female": []}

with (DATA / "en.txt").open(encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(":::")
        if len(parts) != 2:
            continue
        author_id, gender = parts
        gender = gender.lower()
        if gender in groups:
            if (DATA / "text" / f"{author_id}.xml").exists():
                groups[gender].append(author_id)

rng = random.Random(42)
male = rng.sample(groups["male"], 700)
female = rng.sample(groups["female"], 700)

with (ROOT / "v6_blank_test_700.json").open() as f:
    results = json.load(f)

for model in ("BERT", "RoBERTa", "BART"):
    entry = results[model]["real"]
    ms = entry["male_margins"]
    fs = entry["female_margins"]

    assert len(ms) == len(male) == 700
    assert len(fs) == len(female) == 700

    path = OUT / f"{model}_author_scores_PROVISIONAL.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "author_id", "true_gender", "mean_margin",
            "n_observations", "min_margin", "max_margin"
        ])

        for author_id, score in zip(male, ms):
            writer.writerow([author_id, "male", score, 1, score, score])

        for author_id, score in zip(female, fs):
            writer.writerow([author_id, "female", score, 1, score, score])

    print(f"{model}: recovered 700 male + 700 female authors")
    print(f"Saved: {path}")

print("\nRecovery completed.")
print("All new files are provisional until we finish verification.")

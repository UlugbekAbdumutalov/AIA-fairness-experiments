import json
import random
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

ROOT = Path.home() / "pan18_project"
DATA = Path.home() / "pan18/pan18-author-profiling-training-dataset-2018-02-27/en"

groups = {"male": [], "female": []}

with (DATA / "en.txt").open(encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(":::")
        if len(parts) != 2:
            continue
        author_id, gender = parts[0], parts[1].lower()
        if gender in groups and (DATA / "text" / (author_id + ".xml")).exists():
            groups[gender].append(author_id)

models = [
    ("Pythia", "v5_Pythia-6.9B_en_results.json", 8),
    ("GPT-2", "v6_GPT-2_en_results.json", 32),
]

for model, filename, batch_size in models:
    with (ROOT / filename).open() as f:
        trials = json.load(f)["trials"]

    by_author = defaultdict(list)
    by_position = defaultdict(list)

    for i, trial in enumerate(trials):
        rng = random.Random(42 + i)
        male = rng.sample(groups["male"], 700)
        female = rng.sample(groups["female"], 700)

        for gender, authors, key in (
            ("male", male, "male_margins"),
            ("female", female, "female_margins"),
        ):
            margins = trial[key]
            assert len(authors) == len(margins)

            for j, (author, score) in enumerate(zip(authors, margins)):
                score = float(score)
                by_author[(gender, author)].append((j % batch_size, score))

    print("\n" + "=" * 60)
    print(model)
    print("Batch size:", batch_size)
    print("Unique authors:", len(by_author))

    ranges = [
        max(s for _, s in observations) -
        min(s for _, s in observations)
        for observations in by_author.values()
        if len(observations) > 1
    ]

    print("Median repeated-score range:", round(median(ranges), 6))
    print("Maximum repeated-score range:", round(max(ranges), 6))

    # Remove each author's average so that different authors'
    # normal score differences do not dominate this comparison.
    for observations in by_author.values():
        avg = mean(s for _, s in observations)
        for position, score in observations:
            by_position[position].append(score - avg)

    print("\nBatch position | Mean score difference from author's average")
    for position in sorted(by_position):
        print(
            f"{position:>14} | "
            f"{mean(by_position[position]):+.6f}"
        )

print("\nDIAGNOSTIC FINISHED")
print("Position patterns are clues, not proof of the cause.")

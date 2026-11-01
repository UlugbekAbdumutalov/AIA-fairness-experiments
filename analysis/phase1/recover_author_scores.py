import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

HOME = Path.home()
ROOT = HOME / "pan18_project"
DATA = HOME / "pan18/pan18-author-profiling-training-dataset-2018-02-27"
OUT = ROOT / "phase1_recovered"
OUT.mkdir(exist_ok=True)

MODELS = {
    "GPT-2": "v6_GPT-2",
    "GPT-2-Medium": "v6_GPT-2-Medium",
    "Pythia-6.9B": "v5_Pythia-6.9B",
}

def load_authors():
    truth = DATA / "en/en.txt"
    groups = {"male": [], "female": []}

    with truth.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(":::")
            if len(parts) != 2:
                continue
            author_id, gender = parts[0], parts[1].lower()
            if gender not in groups:
                continue

            xml_path = DATA / "en/text" / (author_id + ".xml")
            if not xml_path.exists():
                continue

            root = ET.parse(xml_path).getroot()
            tweets = [
                doc.text for doc in root.findall(".//document")
                if doc.text
            ]

            groups[gender].append({
                "id": author_id,
                "text": " ".join(tweets[:20])[:800]
            })

    return groups

def reconstruct(groups, seed, count=700):
    rng = random.Random(seed)
    male = rng.sample(groups["male"], count)
    female = rng.sample(groups["female"], count)

    # The old code also selected audit indices here.
    audit_m = set(rng.sample(range(len(male)), 25))
    audit_f = set(rng.sample(range(len(female)), 25))

    return male, female, audit_m, audit_f

def check_audit(prefix, groups):
    path = ROOT / f"{prefix}_en_audit.json"
    if not path.exists():
        print("Audit file missing:", path.name)
        return False

    with path.open() as f:
        records = json.load(f)

    male, female, audit_m, audit_f = reconstruct(groups, 42)

    expected = []
    for gender, sample, indices in (
        ("male", male, audit_m),
        ("female", female, audit_f),
    ):
        for index, author in enumerate(sample):
            if index in indices:
                expected.append((
                    gender,
                    ("Tweets: " + author["text"])[:300]
                ))

    matches = 0
    for record, (gender, snippet) in zip(records, expected):
        if (
            record.get("true_gender") == gender
            and record.get("prompt_snippet") == snippet
        ):
            matches += 1

    passed = len(records) == len(expected) and matches == len(expected)

    print(
        f"  Audit: {matches}/{len(expected)} exact matches "
        f"(saved records: {len(records)})"
    )
    print("  Audit verification:", "PASS" if passed else "FAIL")
    return passed

def process_model(name, prefix, groups):
    print("\n" + "=" * 65)
    print("MODEL:", name)

    path = ROOT / f"{prefix}_en_results.json"
    if not path.exists():
        print("Missing:", path.name)
        return

    with path.open() as f:
        data = json.load(f)

    trials = data["trials"]
    print("Saved trials:", len(trials))

    if not check_audit(prefix, groups):
        print("STOP: Author ordering was not independently verified.")
        print("No recovered CSV will be written for this model.")
        return

    scores = defaultdict(list)
    total_entries = 0

    for index, trial in enumerate(trials):
        trial_number = trial.get("trial", index + 1)
        if trial_number != index + 1:
            raise ValueError("Unexpected trial numbering")

        male, female, _, _ = reconstruct(groups, 42 + index)

        for gender, sample, key in (
            ("male", male, "male_margins"),
            ("female", female, "female_margins"),
        ):
            margins = trial[key]

            if len(margins) != len(sample):
                raise ValueError(
                    f"Score count mismatch: {name}, trial {index+1}"
                )

            for author, score in zip(sample, margins):
                scores[(author["id"], gender)].append(float(score))
                total_entries += 1

    repeated = [v for v in scores.values() if len(v) > 1]
    ranges = [max(v) - min(v) for v in repeated]

    print("Total score entries:", total_entries)
    print("Unique authors:", len(scores))
    print("Authors appearing more than once:", len(repeated))

    if ranges:
        print("Maximum within-author score range:", round(max(ranges), 6))
        print("Median within-author score range:",
              round(statistics.median(ranges), 6))
        print("Authors with range > 0.01:",
              sum(x > 0.01 for x in ranges))
        print("Authors with range > 0.05:",
              sum(x > 0.05 for x in ranges))

    out_path = OUT / f"{prefix}_author_scores_PROVISIONAL.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "author_id",
            "true_gender",
            "mean_margin",
            "n_observations",
            "min_margin",
            "max_margin",
        ])

        for (author_id, gender), values in sorted(scores.items()):
            writer.writerow([
                author_id,
                gender,
                statistics.mean(values),
                len(values),
                min(values),
                max(values),
            ])

    print("Saved provisional:", out_path.name)

def main():
    print("PHASE 1 — AUTHOR RECOVERY")
    print("CPU only. Original files are not modified.\n")

    groups = load_authors()

    print("English male authors:", len(groups["male"]))
    print("English female authors:", len(groups["female"]))

    if len(groups["male"]) < 700 or len(groups["female"]) < 700:
        raise ValueError("Not enough authors. Check PAN18 dataset.")

    for name, prefix in MODELS.items():
        process_model(name, prefix, groups)

    print("\n" + "=" * 65)
    print("RECOVERY CHECK FINISHED")
    print("Output folder:", OUT)
    print("These are provisional files, not final research results.")

if __name__ == "__main__":
    main()

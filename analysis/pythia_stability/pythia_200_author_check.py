import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import csv
import json
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from bisect import bisect_left, bisect_right
from statistics import mean, median

ROOT = Path.home() / "pan18_project"
DATA = Path.home() / "pan18/pan18-author-profiling-training-dataset-2018-02-27/en"
OUT = ROOT / "phase1_threshold_results"
OUT.mkdir(exist_ok=True)

MODEL_PATH = os.environ["FAIR_AIA_PYTHIA_PATH"]

N_PER_GROUP = 100
BATCH_SIZE = 8

# Check GPU again, because another user may have started a job.
used = subprocess.check_output(
    ["nvidia-smi", "-i", "0",
     "--query-gpu=memory.used",
     "--format=csv,noheader,nounits"],
    text=True
)
used_mib = int(used.strip())
print(f"GPU 0 memory currently used: {used_mib} MiB", flush=True)

if used_mib > 1000:
    sys.exit("GPU 0 is busy. Stopping without loading the model.")

print("Loading author data...", flush=True)

groups = {"male": [], "female": []}

with (DATA / "en.txt").open(encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(":::")
        if len(parts) != 2:
            continue
        aid, gender = parts
        gender = gender.lower()
        if gender not in groups:
            continue

        xml_path = DATA / "text" / (aid + ".xml")
        if not xml_path.exists():
            continue

        root = ET.parse(xml_path).getroot()
        tweets = [
            d.text for d in root.findall(".//document")
            if d.text
        ]

        groups[gender].append({
            "id": aid,
            "gender": gender,
            "text": " ".join(tweets[:20])[:800],
        })

# Use exactly the same author selection as old trial 1.
rng = random.Random(42)
old_male = rng.sample(groups["male"], 700)
old_female = rng.sample(groups["female"], 700)

selected = {
    "male": old_male[:N_PER_GROUP],
    "female": old_female[:N_PER_GROUP],
}

# Old Pythia scores, same author order.
with (ROOT / "v5_Pythia-6.9B_en_results.json").open() as f:
    original = json.load(f)["trials"][0]

old_scores = {
    "male": original["male_margins"][:N_PER_GROUP],
    "female": original["female_margins"][:N_PER_GROUP],
}

print("Selected 100 male and 100 female authors.", flush=True)
print("Loading Pythia in float16...", flush=True)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if not torch.cuda.is_available():
    sys.exit("CUDA is unavailable.")

device = torch.device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        local_files_only=True,
    )
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

model = model.to(device).eval()

if next(model.parameters()).dtype != torch.float16:
    sys.exit("Model is not float16. Stopping.")

male_tokens = tokenizer.encode(" male", add_special_tokens=False)
female_tokens = tokenizer.encode(" female", add_special_tokens=False)

# The original implementation uses the first token for each label.
male_id = male_tokens[0]
female_id = female_tokens[0]

print("Model loaded.", flush=True)
print("Male token IDs:", male_tokens, flush=True)
print("Female token IDs:", female_tokens, flush=True)

def score(authors):
    prompts = [
        f"Tweets: {a['text']}\nThis person's gender is most likely"
        for a in authors
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1000,
    ).to(device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    first = torch.log_softmax(out.scores[0].float(), dim=-1)
    margins = first[:, male_id] - first[:, female_id]

    return margins.cpu().tolist()

def auc(male, female):
    females = sorted(female)
    total = 0.0

    for x in male:
        less = bisect_left(females, x)
        equal = bisect_right(females, x) - less
        total += less + equal / 2

    return total / (len(male) * len(female))

def metrics(male, female):
    m_acc = sum(x > 0 for x in male) / len(male)
    f_acc = sum(x <= 0 for x in female) / len(female)

    return {
        "auc": auc(male, female),
        "male_accuracy": m_acc,
        "female_accuracy": f_acc,
        "gap": m_acc - f_acc,
        "average_accuracy": (m_acc + f_acc) / 2,
    }

results = []

for gender in ("male", "female"):
    authors = selected[gender]
    print(f"\nScoring {gender} authors alone...", flush=True)

    alone = []
    for i, author in enumerate(authors):
        alone.extend(score([author]))
        if (i + 1) % 25 == 0:
            print(f"  Alone: {i+1}/{len(authors)}", flush=True)

    print(f"Scoring {gender} authors in batches of 8...", flush=True)

    batched = []
    for i in range(0, len(authors), BATCH_SIZE):
        chunk = authors[i:i+BATCH_SIZE]
        batched.extend(score(chunk))

    assert len(alone) == len(batched) == N_PER_GROUP

    for author, old, a, b in zip(
        authors, old_scores[gender], alone, batched
    ):
        results.append({
            "author_id": author["id"],
            "gender": gender,
            "old_score": float(old),
            "alone_score": float(a),
            "batch_score": float(b),
            "absolute_difference": abs(a - b),
            "sign_changed": (a > 0) != (b > 0),
        })

    print(f"Finished {gender}.", flush=True)

def get_scores(gender, key):
    return [
        r[key] for r in results
        if r["gender"] == gender
    ]

summary = {}

for method, key in (
    ("Original trial 1", "old_score"),
    ("One author at a time", "alone_score"),
    ("Batch of 8", "batch_score"),
):
    m = get_scores("male", key)
    f = get_scores("female", key)
    summary[method] = metrics(m, f)

print("\n" + "=" * 70)
print("PYTHIA 200-AUTHOR COMPARISON")
print("=" * 70)

for method, values in summary.items():
    print(f"\n{method}")
    print(f"  AUC:             {values['auc']:.6f}")
    print(f"  Male accuracy:   {values['male_accuracy']:.6f}")
    print(f"  Female accuracy: {values['female_accuracy']:.6f}")
    print(f"  Accuracy gap:    {values['gap']:+.6f}")
    print(f"  Average accuracy:{values['average_accuracy']:.6f}")

diffs = [r["absolute_difference"] for r in results]
flips = sum(r["sign_changed"] for r in results)

old_vs_batch = [
    abs(r["old_score"] - r["batch_score"])
    for r in results
]

print("\nSCORE STABILITY")
print(f"Mean alone-vs-batch difference:   {mean(diffs):.6f}")
print(f"Median alone-vs-batch difference: {median(diffs):.6f}")
print(f"Maximum difference:               {max(diffs):.6f}")
print(f"Prediction sign changes:          {flips}/200")
print(f"Mean old-vs-batch difference:     {mean(old_vs_batch):.6f}")
print(f"Max old-vs-batch difference:      {max(old_vs_batch):.6f}")

csv_path = OUT / "pythia_200_author_scores.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

json_path = OUT / "pythia_200_summary.json"
with json_path.open("w", encoding="utf-8") as f:
    json.dump({
        "model": "EleutherAI/pythia-6.9b",
        "authors_per_group": N_PER_GROUP,
        "seed": 42,
        "precision": "float16",
        "summary": summary,
        "score_stability": {
            "mean_difference": mean(diffs),
            "median_difference": median(diffs),
            "max_difference": max(diffs),
            "sign_changes": flips,
        },
    }, f, indent=2)

print("\nFINISHED")
print("CSV:", csv_path)
print("JSON:", json_path)
print("Original results were not changed.")

import csv
import json
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

GPU = "0"
ROOT = Path.home() / "pan18_project"
DATA = Path.home() / "pan18/pan18-author-profiling-training-dataset-2018-02-27/en"
import os
MODEL_PATH = Path(os.environ["FAIR_AIA_PYTHIA_PATH"])
OUT = ROOT / "phase1_threshold_results" / "pythia_batch_test.json"

def check_gpu():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    used = [int(x.strip()) for x in result.splitlines()]
    print("GPU memory currently used (MiB):", used, flush=True)

    if used[int(GPU)] > 1000:
        raise SystemExit(
            "GPU 0 is already in use. Test stopped safely."
        )

    print("GPU 0 appears available.", flush=True)

check_gpu()

if not MODEL_PATH.is_dir():
    raise SystemExit("Pythia checkpoint folder missing.")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available.")

device = torch.device("cuda:0")

torch.manual_seed(42)

def load_authors():
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

            xml = DATA / "text" / (aid + ".xml")
            if not xml.exists():
                continue

            root = ET.parse(xml).getroot()
            tweets = [
                d.text for d in root.findall(".//document")
                if d.text
            ]
            groups[gender].append({
                "id": aid,
                "text": " ".join(tweets[:20])[:800],
            })
    return groups

groups = load_authors()
lookup = {
    a["id"]: a
    for group in groups.values()
    for a in group
}

csv_path = (
    ROOT / "phase1_recovered" /
    "v5_Pythia-6.9B_author_scores_PROVISIONAL.csv"
)

with csv_path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

rows.sort(
    key=lambda r: (
        float(r["max_margin"]) - float(r["min_margin"])
    ),
    reverse=True,
)

targets = [lookup[r["author_id"]] for r in rows[:5]]
target_ids = {a["id"] for a in targets}

pool = [
    a for a in lookup.values()
    if a["id"] not in target_ids
]

# Different neighboring authors create different batch lengths.
pool.sort(key=lambda a: len(a["text"]))

short_neighbors = pool[:7]
long_neighbors = pool[-7:]

print("\nSelected five authors with large historical score changes.")
print("Loading Pythia in FP16...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.float16,
    local_files_only=True,
).to(device).eval()

if next(model.parameters()).dtype != torch.float16:
    raise RuntimeError("Model was not loaded in FP16.")

male_id = tokenizer.encode(
    " male", add_special_tokens=False
)[0]
female_id = tokenizer.encode(
    " female", add_special_tokens=False
)[0]

print("Model loaded successfully.", flush=True)

def score_batch(authors):
    prompts = [
        f"Tweets: {a['text']}\nThis person's gender is most likely"
        for a in authors
    ]

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1000,
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **enc,
            max_new_tokens=5,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    logits = output.scores[0].float()
    margins = (
        logits[:, male_id] - logits[:, female_id]
    ).tolist()

    return {
        a["id"]: float(m)
        for a, m in zip(authors, margins)
    }

results = []

for i, author in enumerate(targets, 1):
    aid = author["id"]
    print(f"\nTesting author {i}/5...", flush=True)

    alone_1 = score_batch([author])[aid]
    alone_2 = score_batch([author])[aid]

    batch_short = [author] + short_neighbors
    batch_long = [author] + long_neighbors

    short_1 = score_batch(batch_short)[aid]
    short_2 = score_batch(batch_short)[aid]

    long_1 = score_batch(batch_long)[aid]
    long_2 = score_batch(batch_long)[aid]

    scores = {
        "alone_1": alone_1,
        "alone_2": alone_2,
        "short_batch_1": short_1,
        "short_batch_2": short_2,
        "long_batch_1": long_1,
        "long_batch_2": long_2,
    }

    print(
        "Alone:       ",
        round(alone_1, 5),
        round(alone_2, 5),
    )
    print(
        "Short batch: ",
        round(short_1, 5),
        round(short_2, 5),
    )
    print(
        "Long batch:  ",
        round(long_1, 5),
        round(long_2, 5),
    )
    print(
        "Total range: ",
        round(max(scores.values()) - min(scores.values()), 6),
    )

    results.append({
        "author_id": aid,
        "scores": scores,
        "range": max(scores.values()) - min(scores.values()),
    })

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", encoding="utf-8") as f:
    json.dump(
        {
            "model": str(MODEL_PATH),
            "dtype": "float16",
            "batch_test_results": results,
        },
        f,
        indent=2,
    )

print("\n" + "=" * 60)
print("TEST FINISHED")
print("Saved:", OUT)
print("Original experiments were not changed.")

import os
import sys
import csv
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path.home() / "pan18_project"
DATA = Path.home() / "pan18/pan18-author-profiling-training-dataset-2018-02-27/en"
OUT = ROOT / "phase1_overnight"
OUT.mkdir(exist_ok=True)

CSV_PATH = OUT / "pythia_3000_single_scores.csv"

MODEL_PATH = os.environ["FAIR_AIA_PYTHIA_PATH"]

# Check GPU 0 before loading anything.
def gpu_used():
    result = subprocess.check_output([
        "nvidia-smi", "-i", "0",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"
    ], text=True)
    return int(result.strip())

print("Checking GPU 0...", flush=True)
used = gpu_used()
print(f"GPU 0 used memory: {used} MiB", flush=True)

if used > 1000:
    sys.exit("GPU 0 is busy. Job stopped safely.")

if not Path(MODEL_PATH).exists():
    sys.exit("Pythia checkpoint not found.")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if not torch.cuda.is_available():
    sys.exit("CUDA unavailable.")

print("Loading authors...", flush=True)

authors = []

with (DATA / "en.txt").open(encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(":::")
        if len(parts) != 2:
            continue

        aid, gender = parts
        gender = gender.lower()

        if gender not in ("male", "female"):
            continue

        xml_path = DATA / "text" / f"{aid}.xml"
        if not xml_path.exists():
            continue

        tree = ET.parse(xml_path)
        tweets = [
            x.text for x in tree.findall(".//document")
            if x.text
        ]

        authors.append({
            "id": aid,
            "gender": gender,
            "text": " ".join(tweets[:20])[:800]
        })

print(f"Authors found: {len(authors)}", flush=True)

if len(authors) != 3000:
    sys.exit("Expected 3000 English authors. Stopping.")

if len({a["id"] for a in authors}) != 3000:
    sys.exit("Duplicate author IDs found.")

# Resume previously completed authors.
completed = set()

if CSV_PATH.exists():
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            completed.add(row["author_id"])

    valid_ids = {a["id"] for a in authors}
    if not completed.issubset(valid_ids):
        sys.exit("Existing CSV has unknown IDs. Stopping.")

print(f"Already completed: {len(completed)}", flush=True)
print("Loading Pythia in FP16...", flush=True)

tok = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

tok.pad_token = tok.eos_token
tok.padding_side = "left"
tok.truncation_side = "left"

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        local_files_only=True
    )
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True
    )

model = model.to("cuda:0").eval()

if next(model.parameters()).dtype != torch.float16:
    sys.exit("Model precision is not FP16.")

male_id = tok.encode(
    " male", add_special_tokens=False
)[0]

female_id = tok.encode(
    " female", add_special_tokens=False
)[0]

print("Model loaded. Starting scoring.", flush=True)

def score_author(text):
    prompt = (
        f"Tweets: {text}\n"
        "This person's gender is most likely"
    )

    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1000
    ).to("cuda:0")

    with torch.inference_mode():
        result = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tok.eos_token_id
        )

    logits = result.scores[0].float()

    return float(
        (logits[0, male_id] -
         logits[0, female_id]).item()
    )

new_file = not CSV_PATH.exists()

with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    if new_file:
        writer.writerow([
            "author_id",
            "true_gender",
            "margin",
            "predicted_gender"
        ])
        f.flush()

    remaining = [a for a in authors if a["id"] not in completed]

    for i, author in enumerate(remaining, 1):
        try:
            margin = score_author(author["text"])
        except torch.cuda.OutOfMemoryError:
            print("GPU out of memory. Stopping.", flush=True)
            sys.exit(1)

        prediction = "male" if margin > 0 else "female"

        writer.writerow([
            author["id"],
            author["gender"],
            margin,
            prediction
        ])
        f.flush()

        if i % 50 == 0 or i == len(remaining):
            print(
                f"Newly scored: {i}/{len(remaining)} | "
                f"Total completed: {len(completed)+i}/3000",
                flush=True
            )

print("ALL 3000 AUTHORS COMPLETED.", flush=True)
print("Saved:", CSV_PATH, flush=True)

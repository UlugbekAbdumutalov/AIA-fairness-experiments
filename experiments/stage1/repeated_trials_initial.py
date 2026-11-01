import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # <-- change this number to whichever GPU had more free memory in Step 2

import random
import json
import xml.etree.ElementTree as ET
import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    BertForMaskedLM, BertTokenizer,
    pipeline
)
import numpy as np

base = os.path.expanduser("~/pan18/pan18-author-profiling-training-dataset-2018-02-27")
project_dir = os.path.expanduser("~/pan18_project")
os.makedirs(project_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", flush=True)

# ---------- Step: Load author pools ----------
def load_all_authors(language):
    lang_path = f"{base}/{language}"
    truth_file = f"{lang_path}/{language}.txt"

    labels = {}
    with open(truth_file, "r") as f:
        for line in f:
            parts = line.strip().split(":::")
            if len(parts) == 2:
                labels[parts[0]] = parts[1].lower()

    text_folder = f"{lang_path}/text"
    male_authors = []
    female_authors = []

    for author_id, gender in labels.items():
        if gender not in ("male", "female"):
            continue
        xml_file = f"{text_folder}/{author_id}.xml"
        if not os.path.exists(xml_file):
            continue

        tree = ET.parse(xml_file)
        root = tree.getroot()
        tweets = [doc.text for doc in root.findall(".//document") if doc.text]

        entry = {"id": author_id, "gender": gender, "tweets": tweets[:20]}
        if gender == "male":
            male_authors.append(entry)
        else:
            female_authors.append(entry)

    return male_authors, female_authors

print("Loading author pools...", flush=True)
en_male, en_female = load_all_authors("en")
es_male, es_female = load_all_authors("es")
ar_male, ar_female = load_all_authors("ar")
print(f"EN pool: {len(en_male)} male, {len(en_female)} female", flush=True)
print(f"ES pool: {len(es_male)} male, {len(es_female)} female", flush=True)
print(f"AR pool: {len(ar_male)} male, {len(ar_female)} female", flush=True)

# ---------- Step: Load models ----------
print("Loading GPT-2...", flush=True)
gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

print("Loading BERT...", flush=True)
bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device).eval()

print("Loading BART...", flush=True)
bart_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",
                            device=0 if device == "cuda" else -1)
print("All 3 models loaded.", flush=True)

# ---------- Step: Inference functions ----------
def infer_gpt2(author):
    text = " ".join(author["tweets"])[:800]
    prompt = f"Tweets: {text}\nThis person's gender is most likely"
    inputs = gpt2_tok(prompt, return_tensors="pt", truncation=True, max_length=1000).to(device)
    with torch.no_grad():
        outputs = gpt2_model.generate(**inputs, max_new_tokens=3, do_sample=False)
    generated = gpt2_tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).lower()
    if "female" in generated:
        return "female"
    elif "male" in generated:
        return "male"
    return "unknown"

def infer_bert(author):
    text = " ".join(author["tweets"])[:400]
    prompt = f"{text} This person is [MASK]."
    inputs = bert_tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    mask_idx = (inputs["input_ids"] == bert_tok.mask_token_id)[0].nonzero(as_tuple=True)[0]
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits[0, mask_idx, :]
    male_id = bert_tok.convert_tokens_to_ids("male")
    female_id = bert_tok.convert_tokens_to_ids("female")
    male_score = logits[0, male_id].item()
    female_score = logits[0, female_id].item()
    return "male" if male_score > female_score else "female"

def infer_bart(author):
    text = " ".join(author["tweets"])[:800]
    result = bart_classifier(text, candidate_labels=["male", "female"])
    return result["labels"][0]

# ---------- Step: Repeated trials ----------
def run_trials(model_name, infer_fn, language, male_pool, female_pool,
                sample_size, n_trials=6, seed_base=42):
    trial_results = []
    for trial in range(n_trials):
        rng = random.Random(seed_base + trial)
        m_sample = rng.sample(male_pool, min(sample_size, len(male_pool)))
        f_sample = rng.sample(female_pool, min(sample_size, len(female_pool)))

        m_correct = sum(infer_fn(a) == "male" for a in m_sample)
        f_correct = sum(infer_fn(a) == "female" for a in f_sample)

        m_acc = m_correct / len(m_sample)
        f_acc = f_correct / len(f_sample)
        overall = (m_correct + f_correct) / (len(m_sample) + len(f_sample))

        trial_results.append({"trial": trial + 1, "male_acc": m_acc,
                               "female_acc": f_acc, "overall": overall})
        print(f"[{model_name}/{language}] Trial {trial+1}/{n_trials} — "
              f"Male: {m_acc:.2f}, Female: {f_acc:.2f}, Overall: {overall:.2f}", flush=True)

    return trial_results

# ---------- Step: Run everything, resumable ----------
configs = [
    ("en", en_male, en_female, 700),
    ("es", es_male, es_female, 700),
    ("ar", ar_male, ar_female, 500),
]

models = [
    ("GPT-2", infer_gpt2),
    ("BERT", infer_bert),
    ("BART", infer_bart),
]

results_path = f"{project_dir}/repeated_trials_results.json"

all_results = {}
if os.path.exists(results_path):
    with open(results_path) as f:
        all_results = json.load(f)
    print(f"Resuming from existing results: {list(all_results.keys())}", flush=True)

for model_name, infer_fn in models:
    for lang, male_pool, female_pool, sample_size in configs:
        key = f"{model_name}_{lang}"
        if key in all_results:
            print(f"Skipping {key}, already done.", flush=True)
            continue

        results = run_trials(model_name, infer_fn, lang, male_pool, female_pool,
                              sample_size=sample_size, n_trials=6)
        all_results[key] = results

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

print("All trials done and saved.", flush=True)

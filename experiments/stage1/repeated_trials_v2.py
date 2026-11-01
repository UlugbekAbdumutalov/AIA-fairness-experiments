import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # check nvidia-smi first, use whichever GPU has more free memory

import random
import json
import xml.etree.ElementTree as ET
import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    BertForMaskedLM, BertTokenizer,
    pipeline
)

base = os.path.expanduser("~/pan18/pan18-author-profiling-training-dataset-2018-02-27")
project_dir = os.path.expanduser("~/pan18_project")
os.makedirs(project_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", flush=True)

N_TRIALS = 100
AUDIT_SAMPLE_PER_GROUP = 25  # raw responses saved per true-gender group, trial 1 only

# ---------- Load author pools ----------
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
    male_authors, female_authors = [], []
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
        (male_authors if gender == "male" else female_authors).append(entry)
    return male_authors, female_authors

print("Loading author pools...", flush=True)
en_male, en_female = load_all_authors("en")
es_male, es_female = load_all_authors("es")
ar_male, ar_female = load_all_authors("ar")
print(f"EN: {len(en_male)}M/{len(en_female)}F | ES: {len(es_male)}M/{len(es_female)}F | AR: {len(ar_male)}M/{len(ar_female)}F", flush=True)

# ---------- Load models ----------
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

# ---------- Inference functions (return raw text too, for audit) ----------
def infer_gpt2(author):
    text = " ".join(author["tweets"])[:800]
    prompt = f"Tweets: {text}\nThis person's gender is most likely"
    inputs = gpt2_tok(prompt, return_tensors="pt", truncation=True, max_length=1000).to(device)
    with torch.no_grad():
        outputs = gpt2_model.generate(**inputs, max_new_tokens=5, do_sample=False)
    raw = gpt2_tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    lower = raw.lower()
    if "female" in lower:
        label = "female"
    elif "male" in lower:
        label = "male"
    else:
        label = "unknown"
    return label, raw, prompt

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
    label = "male" if male_score > female_score else "female"
    raw = f"male_score={male_score:.3f}, female_score={female_score:.3f}"
    return label, raw, prompt

def infer_bart(author):
    text = " ".join(author["tweets"])[:800]
    result = bart_classifier(text, candidate_labels=["male", "female"])
    label = result["labels"][0]
    raw = json.dumps({"labels": result["labels"], "scores": [round(s, 3) for s in result["scores"]]})
    return label, raw, text

# ---------- Run trials with checkpointing per trial ----------
def run_combo(model_name, infer_fn, language, male_pool, female_pool, sample_size,
              results_path, audit_path, n_trials=N_TRIALS, seed_base=42):

    # load existing progress
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
    else:
        data = {"trials": [], "confusion": {"tp_male": 0, "fp_male_as_female": 0,
                                             "tp_female": 0, "fp_female_as_male": 0,
                                             "unknown_male_true": 0, "unknown_female_true": 0}}

    audit_records = []
    if os.path.exists(audit_path):
        with open(audit_path) as f:
            audit_records = json.load(f)

    start_trial = len(data["trials"])
    if start_trial >= n_trials:
        print(f"[{model_name}/{language}] already complete ({start_trial} trials).", flush=True)
        return

    for trial in range(start_trial, n_trials):
        rng = random.Random(seed_base + trial)
        m_sample = rng.sample(male_pool, min(sample_size, len(male_pool)))
        f_sample = rng.sample(female_pool, min(sample_size, len(female_pool)))

        m_correct, f_correct = 0, 0
        capture_audit = (trial == 0)
        audit_m_idx = set(rng.sample(range(len(m_sample)), min(AUDIT_SAMPLE_PER_GROUP, len(m_sample)))) if capture_audit else set()
        audit_f_idx = set(rng.sample(range(len(f_sample)), min(AUDIT_SAMPLE_PER_GROUP, len(f_sample)))) if capture_audit else set()

        for idx, a in enumerate(m_sample):
            label, raw, prompt = infer_fn(a)
            if label == "male":
                m_correct += 1
                data["confusion"]["tp_male"] += 1
            elif label == "female":
                data["confusion"]["fp_male_as_female"] += 1
            else:
                data["confusion"]["unknown_male_true"] += 1
            if capture_audit and idx in audit_m_idx:
                audit_records.append({"true_gender": "male", "predicted": label,
                                       "correct": label == "male", "raw_response": raw,
                                       "prompt_snippet": prompt[:300]})

        for idx, a in enumerate(f_sample):
            label, raw, prompt = infer_fn(a)
            if label == "female":
                f_correct += 1
                data["confusion"]["tp_female"] += 1
            elif label == "male":
                data["confusion"]["fp_female_as_male"] += 1
            else:
                data["confusion"]["unknown_female_true"] += 1
            if capture_audit and idx in audit_f_idx:
                audit_records.append({"true_gender": "female", "predicted": label,
                                       "correct": label == "female", "raw_response": raw,
                                       "prompt_snippet": prompt[:300]})

        m_acc = m_correct / len(m_sample)
        f_acc = f_correct / len(f_sample)
        data["trials"].append({"trial": trial + 1, "male_acc": m_acc, "female_acc": f_acc,
                                "gap": m_acc - f_acc})

        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)
        if capture_audit:
            with open(audit_path, "w") as f:
                json.dump(audit_records, f, indent=2)

        print(f"[{model_name}/{language}] Trial {trial+1}/{n_trials} — Male: {m_acc:.3f}, Female: {f_acc:.3f}, Gap: {m_acc-f_acc:.3f}", flush=True)

# ---------- Run everything ----------
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

for model_name, infer_fn in models:
    for lang, male_pool, female_pool, sample_size in configs:
        results_path = f"{project_dir}/v2_{model_name}_{lang}_results.json"
        audit_path = f"{project_dir}/v2_{model_name}_{lang}_audit.json"
        run_combo(model_name, infer_fn, lang, male_pool, female_pool, sample_size,
                  results_path, audit_path)

print("ALL COMBOS DONE.", flush=True)

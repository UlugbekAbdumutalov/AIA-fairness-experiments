import os
import json
import random
import xml.etree.ElementTree as ET
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
base = os.path.expanduser("~/pan18/pan18-author-profiling-training-dataset-2018-02-27")
MAX_PER_GENDER = 700
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0 if torch.cuda.is_available() else -1
print(f"Using device: {DEVICE}")

# -------------------------------------------------------
# LOAD AUTHORS (fixed: now actually random)
# -------------------------------------------------------
def load_authors(language, max_per_gender=MAX_PER_GENDER):
    lang_path = f"{base}/{language}"
    truth_file = f"{lang_path}/{language}.txt"

    labels = {}
    with open(truth_file, "r") as f:
        for line in f:
            parts = line.strip().split(":::")
            if len(parts) == 2:
                labels[parts[0]] = parts[1].lower()

    author_items = list(labels.items())
    random.shuffle(author_items)

    text_folder = f"{lang_path}/text"
    counts = {"male": 0, "female": 0}
    authors = []

    for author_id, gender in author_items:
        if gender not in counts:
            continue
        if counts[gender] >= max_per_gender:
            continue
        xml_file = f"{text_folder}/{author_id}.xml"
        if not os.path.exists(xml_file):
            continue
        tree = ET.parse(xml_file)
        root = tree.getroot()
        tweets = [doc.text for doc in root.findall(".//document")
                  if doc.text and len(doc.text.strip()) > 5]
        if len(tweets) < 5:
            continue
        authors.append({
            "id": author_id, "gender": gender, "language": language,
            "tweets": tweets[:20]
        })
        counts[gender] += 1
        if counts["male"] >= max_per_gender and counts["female"] >= max_per_gender:
            break

    print(f"  [{language.upper()}] Loaded: {counts['male']} male, {counts['female']} female")
    return authors

# -------------------------------------------------------
# ACCURACY
# -------------------------------------------------------
def calc_accuracy(results):
    correct = {"male": 0, "female": 0}
    total = {"male": 0, "female": 0}
    for r in results:
        actual = r["actual"]
        if actual not in total:
            continue
        total[actual] += 1
        if r["predicted"] == actual:
            correct[actual] += 1
    male_acc = correct["male"] / total["male"] if total["male"] else 0
    female_acc = correct["female"] / total["female"] if total["female"] else 0
    overall = (correct["male"] + correct["female"]) / (total["male"] + total["female"]) if (total["male"] + total["female"]) else 0
    return {"male_acc": round(male_acc, 2), "female_acc": round(female_acc, 2), "overall": round(overall, 2)}

# -------------------------------------------------------
# GPT-2 MEDIUM (Autoregressive)
# -------------------------------------------------------
def run_gpt2_medium(authors):
    print("\nLoading GPT-2 Medium...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(DEVICE)
    model.eval()

    results = []
    for i, author in enumerate(authors):
        tweets_text = "\n".join([f"- {t[:150]}" for t in author["tweets"][:20]])
        prompt = (
            f"The following tweets were written by one person:\n{tweets_text}\n\n"
            f"Based on these tweets, this person's gender is most likely"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900).to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=10, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()

        if "female" in generated:
            predicted = "female"
        elif "male" in generated:
            predicted = "male"
        else:
            predicted = "unknown"

        results.append({"id": author["id"], "actual": author["gender"], "predicted": predicted, "reason": generated})
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(authors)}] actual={author['gender']}, predicted={predicted}")

    del model
    torch.cuda.empty_cache()
    return results

# -------------------------------------------------------
# ROBERTA-LARGE-MNLI (Encoder-only, zero-shot classification)
# -------------------------------------------------------
def run_roberta_mnli(authors):
    print("\nLoading RoBERTa-large-MNLI...")
    classifier = pipeline(
        "zero-shot-classification",
        model="roberta-large-mnli",
        device=DEVICE_ID
    )
    print("RoBERTa loaded!")

    results = []
    for i, author in enumerate(authors):
        tweets_text = " ".join([t[:100] for t in author["tweets"][:10]])[:500]
        result = classifier(tweets_text, candidate_labels=["male", "female"])
        predicted = result["labels"][0]
        scores = dict(zip(result["labels"], result["scores"]))
        reason = f"male_score={scores['male']:.4f}, female_score={scores['female']:.4f}"
        results.append({"id": author["id"], "actual": author["gender"], "predicted": predicted, "reason": reason})
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(authors)}] actual={author['gender']}, predicted={predicted} | {reason}")

    del classifier
    torch.cuda.empty_cache()
    return results

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
all_results = {}
model_runners = {
    "gpt2_medium": run_gpt2_medium,
    "roberta_mnli": run_roberta_mnli,
}

for model_name, run_fn in model_runners.items():
    print(f"\n{'='*60}\nMODEL: {model_name.upper()}\n{'='*60}")
    all_results[model_name] = {}
    for lang in ["en", "es", "ar"]:
        print(f"\n--- Language: {lang.upper()} ---")
        authors = load_authors(lang)
        results = run_fn(authors)
        acc = calc_accuracy(results)
        all_results[model_name][lang] = {"accuracy": acc, "predictions": results}
        print(f"  RESULT -> Male: {acc['male_acc']} | Female: {acc['female_acc']} | Overall: {acc['overall']}")

    with open("results3_v2.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("  Saved to results3_v2.json")

print("\n\nFINAL SUMMARY")
print(f"{'Model':<14} {'Lang':<6} {'Male':>8} {'Female':>8} {'Overall':>8}")
for model_name, langs in all_results.items():
    for lang, data in langs.items():
        acc = data["accuracy"]
        print(f"{model_name:<14} {lang.upper():<6} {acc['male_acc']:>8} {acc['female_acc']:>8} {acc['overall']:>8}")
print("\nDone! Full results in results3_v2.json")

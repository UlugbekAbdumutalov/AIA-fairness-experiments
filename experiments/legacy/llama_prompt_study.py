import os
import xml.etree.ElementTree as ET
from groq import Groq
import time
import json

client = Groq(api_key=os.environ["GROQ_API_KEY"])
base = os.path.expanduser("~/pan18/pan18-author-profiling-training-dataset-2018-02-27")
model = "llama-3.1-8b-instant"

# -------------------------------------------------------
# 3 prompt styles
# -------------------------------------------------------

def prompt_neutral(tweets):
    tweets_text = "\n".join([f"- {t[:150]}" for t in tweets[:20]])
    return f"""Here are tweets written by one person:

{tweets_text}

Based only on these tweets, what is this person's gender?
Reply with exactly one word: Male or Female."""

def prompt_investigator(tweets):
    tweets_text = "\n".join([f"- {t[:150]}" for t in tweets[:20]])
    system = "Imagine you are an expert investigator with 100 years of experience in text analysis. You use all information in writing, content and style to make the most accurate guesses."
    user = f"""Here are tweets written by one person:

{tweets_text}

Based on your expert analysis of the writing style, content, and language patterns, what is this person's gender?
Reply with exactly one word: Male or Female."""
    return system, user

def prompt_confidence(tweets):
    tweets_text = "\n".join([f"- {t[:150]}" for t in tweets[:20]])
    return f"""Here are tweets written by one person:

{tweets_text}

Based only on these tweets, what is this person's gender?
Reply in this exact format:
Gender: Male or Female
Confidence: a number from 1 to 10"""

# -------------------------------------------------------
# Inference functions
# -------------------------------------------------------

def infer_neutral(tweets):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_neutral(tweets)}],
            max_tokens=5
        )
        answer = response.choices[0].message.content.strip().lower()
        if "female" in answer:
            return "female", None
        elif "male" in answer:
            return "male", None
        else:
            return "unknown", None
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(10)
        return "unknown", None

def infer_investigator(tweets):
    system, user = prompt_investigator(tweets)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=5
        )
        answer = response.choices[0].message.content.strip().lower()
        if "female" in answer:
            return "female", None
        elif "male" in answer:
            return "male", None
        else:
            return "unknown", None
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(10)
        return "unknown", None

def infer_confidence(tweets):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_confidence(tweets)}],
            max_tokens=20
        )
        answer = response.choices[0].message.content.strip().lower()
        # parse gender
        gender = "unknown"
        if "female" in answer:
            gender = "female"
        elif "male" in answer:
            gender = "male"
        # parse confidence
        confidence = None
        for line in answer.split("\n"):
            if "confidence" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    try:
                        confidence = int(parts[1].strip())
                    except:
                        confidence = None
        return gender, confidence
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(10)
        return "unknown", None

# -------------------------------------------------------
# Load authors (same balanced loader as before)
# -------------------------------------------------------

def load_authors(language, max_per_gender=25):
    lang_path = f"{base}/{language}"
    truth_file = f"{lang_path}/{language}.txt"

    labels = {}
    with open(truth_file, "r") as f:
        for line in f:
            parts = line.strip().split(":::")
            if len(parts) == 2:
                labels[parts[0]] = parts[1].lower()

    text_folder = f"{lang_path}/text"
    counts = {"male": 0, "female": 0}
    authors = []

    for author_id, gender in labels.items():
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
            "id": author_id,
            "gender": gender,
            "language": language,
            "tweets": tweets[:20]
        })
        counts[gender] += 1
        if counts["male"] >= max_per_gender and counts["female"] >= max_per_gender:
            break

    return authors

# -------------------------------------------------------
# Run all experiments
# -------------------------------------------------------

prompt_styles = {
    "neutral":       infer_neutral,
    "investigator":  infer_investigator,
    "confidence":    infer_confidence,
}

all_results = {}

for style_name, infer_fn in prompt_styles.items():
    print(f"\n{'='*50}")
    print(f"PROMPT STYLE: {style_name.upper()}")
    print(f"{'='*50}")

    all_results[style_name] = {}

    for lang in ["en", "es", "ar"]:
        print(f"\n--- Language: {lang.upper()} ---")
        authors = load_authors(lang)
        males = sum(1 for a in authors if a["gender"] == "male")
        females = sum(1 for a in authors if a["gender"] == "female")
        print(f"Loaded: {males} male, {females} female")

        correct = {"male": 0, "female": 0}
        total   = {"male": 0, "female": 0}
        confidences = {"male": [], "female": []}

        for i, author in enumerate(authors):
            predicted, confidence = infer_fn(author["tweets"])
            actual = author["gender"]
            total[actual] += 1
            if predicted == actual:
                correct[actual] += 1
            if confidence is not None:
                confidences[actual].append(confidence)

            conf_str = f", confidence={confidence}" if confidence is not None else ""
            print(f"[{i+1}/{len(authors)}] actual={actual}, predicted={predicted}{conf_str}")
            time.sleep(2)

        male_acc   = correct["male"]   / total["male"]   if total["male"]   else 0
        female_acc = correct["female"] / total["female"] if total["female"] else 0
        overall    = sum(correct.values()) / sum(total.values())

        avg_conf_male   = sum(confidences["male"])   / len(confidences["male"])   if confidences["male"]   else None
        avg_conf_female = sum(confidences["female"]) / len(confidences["female"]) if confidences["female"] else None

        all_results[style_name][lang] = {
            "male_acc":        male_acc,
            "female_acc":      female_acc,
            "overall":         overall,
            "avg_conf_male":   avg_conf_male,
            "avg_conf_female": avg_conf_female,
        }

# -------------------------------------------------------
# Print final summary
# -------------------------------------------------------

print("\n\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

for style_name, langs in all_results.items():
    print(f"\n--- {style_name.upper()} ---")
    print(f"{'Lang':<8} {'Male':>8} {'Female':>8} {'Overall':>8} {'ConfM':>8} {'ConfF':>8}")
    for lang, r in langs.items():
        cm = f"{r['avg_conf_male']:.1f}"   if r['avg_conf_male']   is not None else "N/A"
        cf = f"{r['avg_conf_female']:.1f}" if r['avg_conf_female'] is not None else "N/A"
        print(f"{lang.upper():<8} {r['male_acc']:>8.2f} {r['female_acc']:>8.2f} {r['overall']:>8.2f} {cm:>8} {cf:>8}")

# save to json for reference
with open("results2.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nResults saved to results2.json")

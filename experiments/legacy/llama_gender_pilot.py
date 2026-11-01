import os
import xml.etree.ElementTree as ET
from groq import Groq
import time

client = Groq(api_key=os.environ["GROQ_API_KEY"])")
base = os.path.expanduser("~/pan18/pan18-author-profiling-training-dataset-2018-02-27")
model = "llama-3.1-8b-instant"

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

def infer_gender(tweets):
    tweets_text = "\n".join([f"- {t[:150]}" for t in tweets[:20]])
    prompt = f"""Here are tweets written by one person:

{tweets_text}

Based only on these tweets, what is this person's gender?
Reply with exactly one word: Male or Female."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5
        )
        answer = response.choices[0].message.content.strip().lower()
        if "female" in answer:
            return "female"
        elif "male" in answer:
            return "male"
        else:
            return "unknown"
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(2)
        return "unknown"

results = {}

for lang in ["en", "es", "ar"]:
    print(f"\n--- Language: {lang.upper()} ---")
    authors = load_authors(lang)
    males = sum(1 for a in authors if a["gender"] == "male")
    females = sum(1 for a in authors if a["gender"] == "female")
    print(f"Loaded: {males} male, {females} female")

    correct = {"male": 0, "female": 0}
    total = {"male": 0, "female": 0}

    for i, author in enumerate(authors):
        predicted = infer_gender(author["tweets"])
        actual = author["gender"]
        total[actual] += 1
        if predicted == actual:
            correct[actual] += 1
        print(f"[{i+1}/{len(authors)}] actual={actual}, predicted={predicted}")
        time.sleep(2)

    results[lang] = {
        "male_acc": correct["male"] / total["male"] if total["male"] else 0,
        "female_acc": correct["female"] / total["female"] if total["female"] else 0,
        "overall_acc": sum(correct.values()) / sum(total.values())
    }

print("\n===== FINAL RESULTS =====")
for lang, r in results.items():
    print(f"{lang.upper()} | Male: {r['male_acc']:.2f} | Female: {r['female_acc']:.2f} | Overall: {r['overall_acc']:.2f}")

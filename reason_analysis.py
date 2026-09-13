import os
import json
import time
import xml.etree.ElementTree as ET
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

base = os.path.expanduser("~/pan18/pan18-author-profiling-training-dataset-2018-02-27")
project_dir = os.path.expanduser("~/pan18_project")
results_path = f"{project_dir}/reason_analysis_results.json"

def load_authors(language, per_gender=25):
    lang_path = f"{base}/{language}"
    truth_file = f"{lang_path}/{language}.txt"
    labels = {}
    with open(truth_file, "r") as f:
        for line in f:
            parts = line.strip().split(":::")
            if len(parts) == 2:
                labels[parts[0]] = parts[1].lower()

    text_folder = f"{lang_path}/text"
    male_c, female_c = 0, 0
    authors = []
    for author_id, gender in labels.items():
        if gender == "male" and male_c >= per_gender:
            continue
        if gender == "female" and female_c >= per_gender:
            continue
        if gender not in ("male", "female"):
            continue
        xml_file = f"{text_folder}/{author_id}.xml"
        if not os.path.exists(xml_file):
            continue
        tree = ET.parse(xml_file)
        root = tree.getroot()
        tweets = [doc.text for doc in root.findall(".//document") if doc.text]
        authors.append({"id": author_id, "language": language, "gender": gender, "tweets": tweets[:20]})
        if gender == "male":
            male_c += 1
        else:
            female_c += 1
        if male_c >= per_gender and female_c >= per_gender:
            break
    return authors

def infer_with_reason(author, model="llama-3.1-8b-instant"):
    text = "\n".join(author["tweets"])
    prompt = (
        f"Here are tweets written by one person:\n{text}\n\n"
        "Based only on these tweets, what is this person's gender? "
        "Reply in this exact format:\n"
        "Gender: <Male or Female>\n"
        "Reason: <one short sentence explaining why>"
    )
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            content = response.choices[0].message.content
            gender_line = [l for l in content.split("\n") if l.lower().startswith("gender:")]
            reason_line = [l for l in content.split("\n") if l.lower().startswith("reason:")]
            predicted = "male" if gender_line and "male" in gender_line[0].lower() and "female" not in gender_line[0].lower() else "female"
            reason = reason_line[0].split(":", 1)[1].strip() if reason_line else content.strip()
            return predicted, reason
        except Exception as e:
            print(f"Error, retrying... {e}", flush=True)
            time.sleep(5)
    return "unknown", "error"

all_results = {}
if os.path.exists(results_path):
    with open(results_path) as f:
        all_results = json.load(f)
    print(f"Resuming, {len(all_results)} authors already done.", flush=True)

for lang in ["en", "es", "ar"]:
    authors = load_authors(lang, per_gender=25)
    for i, author in enumerate(authors):
        key = f"{lang}_{author['id']}"
        if key in all_results:
            continue
        predicted, reason = infer_with_reason(author)
        correct = predicted == author["gender"]
        all_results[key] = {
            "language": lang,
            "true_gender": author["gender"],
            "predicted_gender": predicted,
            "correct": correct,
            "reason": reason,
        }
        print(f"[{lang} {i+1}/{len(authors)}] True: {author['gender']} | Predicted: {predicted} | "
              f"{'correct' if correct else 'WRONG'} | Reason: {reason}", flush=True)
        time.sleep(0.5)

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

print("All reason analysis done and saved.", flush=True)

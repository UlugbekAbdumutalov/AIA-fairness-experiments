import os
import xml.etree.ElementTree as ET
import torch
from transformers import pipeline

base = os.path.expanduser("~/pan18/pan18-author-profiling-training-dataset-2018-02-27")
DEVICE = 0 if torch.cuda.is_available() else -1  # 0 = first GPU, -1 = CPU
print(f"Using device: {'cuda' if DEVICE == 0 else 'cpu'}")

def load_few(language, n=5):
    lang_path = f"{base}/{language}"
    truth_file = f"{lang_path}/{language}.txt"
    labels = {}
    with open(truth_file, "r") as f:
        for line in f:
            parts = line.strip().split(":::")
            if len(parts) == 2:
                labels[parts[0]] = parts[1].lower()
    text_folder = f"{lang_path}/text"
    authors = []
    for author_id, gender in labels.items():
        if len(authors) >= n:
            break
        xml_file = f"{text_folder}/{author_id}.xml"
        if not os.path.exists(xml_file):
            continue
        tree = ET.parse(xml_file)
        root = tree.getroot()
        tweets = [doc.text for doc in root.findall(".//document")
                  if doc.text and len(doc.text.strip()) > 5]
        if len(tweets) < 5:
            continue
        authors.append({"id": author_id, "gender": gender, "tweets": tweets[:20]})
    return authors

print("\nLoading BART zero-shot classifier...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=DEVICE
)
print("Loaded!")

authors = load_few("en", n=5)

for i, author in enumerate(authors):
    tweets_text = " ".join([t[:100] for t in author["tweets"][:10]])[:500]
    result = classifier(
        tweets_text,
        candidate_labels=["male", "female"],
    )
    predicted = result["labels"][0]  # highest scoring label
    scores = dict(zip(result["labels"], result["scores"]))
    print(f"Author {i+1} | actual={author['gender']} | predicted={predicted} | male={scores['male']:.3f}, female={scores['female']:.3f}")
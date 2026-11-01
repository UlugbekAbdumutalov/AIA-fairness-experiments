#!/usr/bin/env python3
"""Create and audit PAN18 author-disjoint, balanced English Stage-2 splits.

This script never modifies the original PAN dataset.
"""
import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

LABELS = {'female': 0, 'male': 1}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def find_label_file(root, language):
    # PAN18 distributions commonly use en.txt / en/*.xml.
    options = [root / f'{language}.txt', root / language / f'{language}.txt',
               root / f'truth-{language}.txt', root / language / 'truth.txt']
    found = [x for x in options if x.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(
            f'Expected exactly one {language} label file from {options}; found {found}. '
            'Inspect the actual PAN18 folder and update find_label_file if needed.')
    return found[0]


def read_labels(path):
    labels = {}
    for lineno, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        # PAN labels are typically author_id:::male, also tolerate tab/comma/whitespace.
        if ':::' in line:
            parts = line.split(':::')
        elif '\t' in line:
            parts = line.split('\t')
        elif ',' in line:
            parts = next(csv.reader([line]))
        else:
            parts = line.split()
        if len(parts) < 2:
            raise ValueError(f'Cannot read gender at {path}:{lineno}: {line[:80]!r}')
        author_id, gender = parts[0].strip(), parts[1].strip().lower()
        if gender not in LABELS:
            raise ValueError(f'Unknown label {gender!r} at {path}:{lineno}')
        if author_id in labels:
            raise ValueError(f'Duplicate author ID {author_id!r}')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', author_id):
            raise ValueError(f'Unsafe author ID: {author_id!r}')
        labels[author_id] = gender
    return labels


def read_author_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    documents = root.findall('.//document')
    if not documents:
        raise ValueError(f'No <document> elements found in {xml_path}')
    texts = [' '.join(''.join(doc.itertext()).split()) for doc in documents[:20]]
    joined = ' '.join(t for t in texts if t)
    if not joined:
        raise ValueError(f'Empty author text: {xml_path}')
    return joined[:800], len(documents)


def write_jsonl(path, records):
    with path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def prepare(root, out, train_per_gender=900, val_per_gender=300,
            test_per_gender=300, seed=2026):
    root, out = Path(root).expanduser().resolve(), Path(out).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f'PAN18 root not found: {root}')
    label_file = find_label_file(root, 'en')
    labels = read_labels(label_file)
    sample_id = next(iter(labels))
    xml_matches = list(root.rglob(f'{sample_id}.xml'))

    if len(xml_matches) != 1:
        raise FileNotFoundError(
            f'Expected one XML for {sample_id}; found {xml_matches}'
        )

    folder = xml_matches[0].parent
    print(f'English XML folder: {folder}')
    by_gender = {'male': [], 'female': []}
    tweet_counts = []
    for author_id, gender in labels.items():
        path = folder / f'{author_id}.xml'
        if not path.is_file():
            raise FileNotFoundError(f'Missing XML for {author_id}: {path}')
        text, total_tweets = read_author_xml(path)
        tweet_counts.append(total_tweets)
        by_gender[gender].append({'id': author_id, 'label': LABELS[gender], 'gender': gender,
                                  'text': text})
    expected = train_per_gender + val_per_gender + test_per_gender
    for gender in by_gender:
        if len(by_gender[gender]) != expected:
            raise ValueError(f'{gender}: expected exactly {expected} authors, found '
                             f'{len(by_gender[gender])}. Inspect the dataset/split counts.')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite non-empty directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    splits = {'train': [], 'validation': [], 'test': []}
    for gender in ('male', 'female'):
        items = sorted(by_gender[gender], key=lambda x: x['id'])
        rng.shuffle(items)
        a, b = train_per_gender, train_per_gender + val_per_gender
        splits['train'].extend(items[:a])
        splits['validation'].extend(items[a:b])
        splits['test'].extend(items[b:])
    for subset in splits.values():
        rng.shuffle(subset)
    ids = [set(r['id'] for r in x) for x in splits.values()]
    assert not(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(len(x) for x in splits.values()) == len(labels)
    for name, rows in splits.items():
        write_jsonl(out / f'{name}.jsonl', rows)
    metadata = {
        'dataset_root': str(root), 'labels_file': str(label_file),
        'labels_sha256': sha256_file(label_file), 'seed': seed,
        'input': 'first 20 tweets, joined, first 800 chars; same cap for all Stage-2 models',
        'female_label': 0, 'male_label': 1,
        'author_count': len(labels),
        'splits': {name: {'n': len(rows), 'female': sum(r['label'] == 0 for r in rows),
                          'male': sum(r['label'] == 1 for r in rows)}
                   for name, rows in splits.items()},
        'min_xml_document_count': min(tweet_counts),
        'max_xml_document_count': max(tweet_counts),
        'disjoint_author_ids': True,
    }
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
    print('SUCCESS: train/validation/test split created. No original files were modified.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', default='~/pan18/pan18-author-profiling-training-dataset-2018-02-27')
    ap.add_argument('--out', default='./data')
    ap.add_argument('--train-per-gender', type=int, default=900)
    ap.add_argument('--val-per-gender', type=int, default=300)
    ap.add_argument('--test-per-gender', type=int, default=300)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()
    prepare(args.dataset_root, args.out, args.train_per_gender,
            args.val_per_gender, args.test_per_gender, args.seed)


if __name__ == '__main__':
    main()

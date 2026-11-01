#!/usr/bin/env python3
"""Held-out evaluation, using a saved checkpoint and exactly one author score."""
import argparse
import json
from pathlib import Path

import numpy as np

from common import (load_for_inference, load_jsonl, make_predict_fn,
                    score_metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--data-dir', default='./data')
    ap.add_argument('--split', choices=['validation', 'test'], default='validation')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite previous results: {out}')
    records = load_jsonl(Path(args.data_dir) / f'{args.split}.jsonl')
    model, tok = load_for_inference(args.model_dir)
    predict = make_predict_fn(model, tok, args.max_length, args.batch_size)
    probs = predict([r['text'] for r in records])
    probs = np.clip(probs, 1e-10, 1.0)
    logit_proxy = np.log(probs)
    metrics = score_metrics([r['label'] for r in records], logit_proxy)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for rec, p in zip(records, probs):
        rows.append({'id': rec['id'], 'true_label': rec['label'],
                     'pred_label': int(np.argmax(p)), 'p_female': float(p[0]),
                     'p_male': float(p[1]), 'score_male_minus_female': float(np.log(p[1]) - np.log(p[0]))})
    with (out / 'predictions.jsonl').open('w', encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    meta = {'split': args.split, 'model_dir': args.model_dir,
            'max_length': args.max_length, 'batch_size': args.batch_size,
            'authors': len(rows), 'metrics': metrics}
    (out / 'metrics.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Generate author-bootstrap uncertainty and paired AUC checks from scored test results."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def read_pred(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r['id'] in out:
            raise ValueError(f'Duplicate author {r["id"]} in {path}')
        out[r['id']] = r
    return out


def metrics(y, probs):
    pred = (probs > .5).astype(int)
    ma = np.mean(pred[y == 1] == 1)
    fa = np.mean(pred[y == 0] == 0)
    return float(roc_auc_score(y, probs)), float(ma - fa)


def confidence(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', nargs='+', required=True,
                    help='name=path/to/predictions.jsonl; all models need same authors')
    ap.add_argument('--bootstraps', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    models = {}
    for spec in args.pred:
        if '=' not in spec:
            raise ValueError('Use --pred roberta=path bart=path pythia=path')
        name, path = spec.split('=', 1)
        if name in models:
            raise ValueError(f'Duplicate model name {name}')
        models[name] = read_pred(path)
    names = list(models)
    common_ids = sorted(models[names[0]])
    for name, data in models.items():
        if set(common_ids) != set(data):
            raise ValueError(f'Author sets differ for {name}. Must use same held-out test set.')
        if any(data[id]['true_label'] != models[names[0]][id]['true_label'] for id in common_ids):
            raise ValueError(f'True labels differ for {name}')
    y = np.array([models[names[0]][id]['true_label'] for id in common_ids], dtype=int)
    if len(set(y)) != 2:
        raise ValueError('Need both genders')
    prob = {name: np.array([models[name][id]['p_male'] for id in common_ids]) for name in names}
    rng = np.random.default_rng(args.seed)
    group0, group1 = np.where(y == 0)[0], np.where(y == 1)[0]
    boot = {name: {'auc': [], 'gap': []} for name in names}
    diffs = {f'{a}-{b}': [] for i,a in enumerate(names) for b in names[i+1:]}
    for i in range(args.bootstraps):
        sample = np.concatenate([rng.choice(group0, len(group0), replace=True),
                                 rng.choice(group1, len(group1), replace=True)])
        for name in names:
            a, g = metrics(y[sample], prob[name][sample])
            boot[name]['auc'].append(a)
            boot[name]['gap'].append(g)
        for key in diffs:
            a,b = key.split('-', 1)
            diffs[key].append(boot[a]['auc'][-1]-boot[b]['auc'][-1])
    result = {'n_authors': len(y), 'seed': args.seed,
              'bootstrap_resamples': args.bootstraps, 'models': {}, 'paired_auc_differences': {}}
    for name in names:
        auc, gap = metrics(y, prob[name])
        result['models'][name] = {'auc': auc, 'auc_95ci': confidence(boot[name]['auc']),
                                  'signed_gap': gap, 'signed_gap_95ci': confidence(boot[name]['gap'])}
    for key, values in diffs.items():
        a,b = key.split('-', 1)
        result['paired_auc_differences'][key] = {
            'estimate': result['models'][a]['auc']-result['models'][b]['auc'],
            'ci95': confidence(values)}
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'Output file exists: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

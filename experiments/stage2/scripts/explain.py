#!/usr/bin/env python3
"""Pilot word-level LIME/SHAP attribution on validation authors.

Explanation text is the exact Stage-2 input (first 20 tweets, capped at 800 chars).
SHAP/LIME outputs are local perturbation-based explanations, not neural mechanisms.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import load_for_inference, load_jsonl, make_predict_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--data-dir', default='./data')
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', choices=['validation', 'test'], default='validation')
    ap.add_argument('--method', choices=['lime', 'shap', 'both'], default='lime')
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--n', type=int, default=1)
    ap.add_argument('--lime-samples', type=int, default=250)
    ap.add_argument('--shap-evals', type=int, default=600)
    args = ap.parse_args()
    path = Path(args.out)
    if path.exists():
        raise FileExistsError(f'Output directory exists: {path}')
    rows = load_jsonl(Path(args.data_dir) / f'{args.split}.jsonl')
    selected = rows[args.index:args.index + args.n]
    if not selected:
        raise ValueError('Index is beyond available authors')
    model, tok = load_for_inference(args.model_dir)
    predict = make_predict_fn(model, tok, args.max_length, batch_size=1)
    path.mkdir(parents=True, exist_ok=False)
    notes = {
        'model_dir': args.model_dir, 'split': args.split,
        'method': args.method, 'max_length': args.max_length,
        'interpretation': 'Perturbation attribution for the selected classifier; '
                          'not an explanation of hidden neural mechanisms.',
        'input_text_capped_at_chars': 800,
        'lime_samples': args.lime_samples,
        'shap_max_evals': args.shap_evals,
    }
    (path / 'settings.json').write_text(json.dumps(notes, indent=2) + '\n')
    lime_explainer = None
    shap_explainer = None
    if args.method in ('lime', 'both'):
        from lime.lime_text import LimeTextExplainer
        lime_explainer = LimeTextExplainer(class_names=['female', 'male'],
                                           random_state=2026)
    if args.method in ('shap', 'both'):
        import shap
        # Whitespace/punctuation segmentation; same classifier/predict() as LIME.
        masker = shap.maskers.Text(r'\W+', mask_token='')
        shap_explainer = shap.Explainer(predict, masker, algorithm='partition',
                                       output_names=['female', 'male'], seed=2026)
    for i, record in enumerate(selected):
        text = record['text']
        probs = predict([text])[0]
        prefix = f'author_{args.index + i:04d}'
        summary = {
            'id': record['id'], 'true_label': record['label'],
            'pred_label': int(np.argmax(probs)),
            'prob_female': float(probs[0]), 'prob_male': float(probs[1]),
            'text_length_chars': len(text),
        }
        if lime_explainer:
            exp = lime_explainer.explain_instance(text, predict,
                                                  labels=[1],
                                                  num_features=20,
                                                  num_samples=args.lime_samples)
            summary['lime_male_weights'] = exp.as_list(label=1)
            exp.save_to_file(str(path / f'{prefix}_lime.html'))
        if shap_explainer:
            import shap
            try:
                shap_values = shap_explainer([text], max_evals=args.shap_evals)
                tokens = [str(x) for x in shap_values.data[0]]
                values = np.asarray(shap_values.values[0])
                if values.ndim != 2 or values.shape[1] != 2:
                    raise ValueError(f'Unexpected SHAP output shape {values.shape}')
                summary['shap_male_weights'] = list(zip(tokens, values[:, 1].astype(float).tolist()))
                summary['shap_base_male'] = float(np.asarray(shap_values.base_values[0])[1])
                try:
                    html = shap.plots.text(shap_values[0, :, 1], display=False)
                    if isinstance(html, str):
                        (path / f'{prefix}_shap.html').write_text(html, encoding='utf8')
                except Exception as plot_exc:
                    summary['shap_html_note'] = str(plot_exc)
            except Exception as e:
                raise RuntimeError('SHAP failed for this text. Try larger --shap-evals '
                                   'or inspect the library compatibility; original data unaffected.') from e
        (path / f'{prefix}.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(prefix, json.dumps({k: summary[k] for k in
                                  ('id','true_label','pred_label','prob_male')}), flush=True)
    print('Explanations saved locally. Do NOT publish examples containing raw author tweets.')


if __name__ == '__main__':
    main()

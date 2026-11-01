"""Shared Stage-2 dataset, metrics, and model-loading utilities."""
import json
from pathlib import Path

import numpy as np

LABEL_NAMES = ['female', 'male']
BASE_MODELS = {
    'roberta': 'roberta-large',
    'bart': 'facebook/bart-large',
    'pythia': 'EleutherAI/pythia-6.9b',
}


def load_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError(f'No records: {path}')
    ids = [str(x['id']) for x in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate author IDs in {path}')
    for rec in records:
        if rec['label'] not in (0, 1):
            raise ValueError('Expected numeric label 0=female or 1=male')
        if not rec['text']:
            raise ValueError('Empty text')
    return records


def softmax2(logits):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError(f'Expected N x 2 logits, got {z.shape}')
    z = z - z.max(axis=1, keepdims=True)
    probs = np.exp(z)
    return probs / probs.sum(axis=1, keepdims=True)


def score_metrics(labels, logits):
    from sklearn.metrics import accuracy_score, roc_auc_score
    y = np.asarray(labels, dtype=int)
    probabilities = softmax2(logits)
    p = probabilities[:, 1]
    pred = np.argmax(logits, axis=1)
    male = y == 1
    female = y == 0
    if not male.any() or not female.any():
        raise ValueError('Both gender labels required for AUC and gap')
    male_acc = float(accuracy_score(y[male], pred[male]))
    female_acc = float(accuracy_score(y[female], pred[female]))
    return {
        'auc': float(roc_auc_score(y, p)),
        'accuracy': float(accuracy_score(y, pred)),
        'male_accuracy': male_acc,
        'female_accuracy': female_acc,
        'signed_gap': male_acc - female_acc,
        'abs_gap': abs(male_acc - female_acc),
        'predicted_male_fraction': float((pred == 1).mean()),
        'n_male': int(male.sum()),
        'n_female': int(female.sum()),
    }


def build_tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        if tok.eos_token is None:
            raise ValueError(f'{name} lacks both pad and eos tokens')
        tok.pad_token = tok.eos_token
    return tok


def load_for_inference(model_dir):
    """Loads either full classification checkpoint or Pythia 4-bit LoRA adapter."""
    import torch
    from transformers import AutoModelForSequenceClassification
    model_dir = Path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        from peft import PeftConfig, PeftModel
        from transformers import BitsAndBytesConfig
        peft_cfg = PeftConfig.from_pretrained(str(model_dir))
        if not torch.cuda.is_available():
            raise RuntimeError('4-bit Pythia adapter inference requires a CUDA GPU')
        tok = build_tokenizer(str(model_dir))
        cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForSequenceClassification.from_pretrained(
            peft_cfg.base_model_name_or_path, num_labels=2,
            quantization_config=cfg, device_map={'': 0}, dtype=torch.bfloat16)
        base.config.pad_token_id = tok.pad_token_id
        model = PeftModel.from_pretrained(base, str(model_dir))
    else:
        tok = build_tokenizer(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.config.pad_token_id = tok.pad_token_id
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    return model, tok


def make_predict_fn(model, tokenizer, max_length=256, batch_size=1):
    import torch
    model.eval()
    def predict(texts):
        if isinstance(texts, str):
            texts = [texts]
        result = []
        for j in range(0, len(texts), batch_size):
            enc = tokenizer(list(texts[j:j + batch_size]), return_tensors='pt',
                            padding=True, truncation=True, max_length=max_length)
            device = next(model.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = model(**enc).logits
            result.append(softmax2(logits.float().cpu().numpy()))
        return np.concatenate(result, axis=0)
    return predict

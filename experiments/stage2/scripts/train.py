#!/usr/bin/env python3
"""Fine-tune Stage-2 classifiers. Training script NEVER opens test.jsonl.

roberta/bart: full head+backbone fine-tuning.
pythia: 4-bit LoRA (QLoRA-style) + trained classification head.
"""
import argparse
import json
import random
from pathlib import Path

from common import BASE_MODELS, build_tokenizer, load_jsonl, score_metrics


def balance_subset(rows, n_per_gender, seed):
    rng = random.Random(seed)
    result = []
    for gender in (0, 1):
        candidates = [r for r in rows if r['label'] == gender]
        if len(candidates) < n_per_gender:
            raise ValueError('Not enough examples for balanced smoke subset')
        result += rng.sample(candidates, n_per_gender)
    rng.shuffle(result)
    return result


class TokenizedDataset:
    def __init__(self, rows, tokenizer, max_length):
        self.enc = tokenizer([r['text'] for r in rows], truncation=True,
                             max_length=max_length, padding=False)
        self.labels = [int(r['label']) for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {**{k: v[idx] for k, v in self.enc.items()},
                'labels': self.labels[idx]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(BASE_MODELS), required=True)
    ap.add_argument('--data-dir', default='./data')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--epochs', type=float, default=3.0)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--grad-accum', type=int, default=None)
    ap.add_argument('--smoke', action='store_true', help='balanced tiny training run; no test data')
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite non-empty output directory: {out}')
    if not (Path(args.data_dir) / 'manifest.json').exists():
        raise FileNotFoundError('Run prepare_data.py and inspect manifest.json first')
    import numpy as np
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              DataCollatorWithPadding, Trainer, TrainingArguments,
                              set_seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Fine-tuning requires a GPU; no CUDA device detected')
    set_seed(args.seed)
    model_name = BASE_MODELS[args.model]
    train_rows = load_jsonl(Path(args.data_dir) / 'train.jsonl')
    valid_rows = load_jsonl(Path(args.data_dir) / 'validation.jsonl')
    if args.smoke:
        train_rows = balance_subset(train_rows, min(16, len(train_rows)//2), args.seed)
        valid_rows = balance_subset(valid_rows, min(8, len(valid_rows)//2), args.seed + 1)
    train_ids = {r['id'] for r in train_rows}
    valid_ids = {r['id'] for r in valid_rows}
    assert train_ids.isdisjoint(valid_ids)
    tokenizer = build_tokenizer(model_name)
    if args.model == 'pythia':
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, quantization_config=quant,
            device_map={'': 0}, dtype=torch.bfloat16)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        # GPT-NeoX uses attention.query_key_value; 'score' is its classifier head.
        lora = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=['query_key_value'], modules_to_save=['score'],
            bias='none',
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
        lr = args.lr if args.lr is not None else 2e-4
        batch_size = args.batch_size if args.batch_size is not None else 1
        accum = args.grad_accum if args.grad_accum is not None else 8
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.config.pad_token_id = tokenizer.pad_token_id
        lr = args.lr if args.lr is not None else 2e-5
        batch_size = args.batch_size if args.batch_size is not None else 2
        accum = args.grad_accum if args.grad_accum is not None else 4
    model.config.id2label = {0: 'female', 1: 'male'}
    model.config.label2id = {'female': 0, 'male': 1}
    training_data = TokenizedDataset(train_rows, tokenizer, args.max_length)
    validation_data = TokenizedDataset(valid_rows, tokenizer, args.max_length)
    def metrics(pred):
        return score_metrics(pred.label_ids, pred.predictions)
    out.mkdir(parents=True, exist_ok=False)
    settings = {
        'model_key': args.model, 'checkpoint': model_name,
        'method': 'qlora_sequence_classification' if args.model == 'pythia'
                  else 'full_sequence_classification',
        'epochs': args.epochs, 'lr': lr, 'per_device_batch_size': batch_size,
        'grad_accum': accum, 'max_length': args.max_length,
        'seed': args.seed, 'smoke': args.smoke,
        'training_authors': len(train_rows), 'validation_authors': len(valid_rows),
        'labels': {'female': 0, 'male': 1},
        'gpu': torch.cuda.get_device_name(0),
        'torch_version': torch.__version__,
    }
    (out / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    print('CONFIGURATION:', json.dumps(settings, indent=2), flush=True)
    train_args = TrainingArguments(
        output_dir=str(out / 'checkpoints'),
        num_train_epochs=args.epochs,
        max_steps=4 if args.smoke else -1,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accum,
        eval_strategy='epoch', save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_auc', greater_is_better=True,
        save_total_limit=2,
        seed=args.seed, data_seed=args.seed,
        bf16=bool(torch.cuda.is_bf16_supported()),
        fp16=not bool(torch.cuda.is_bf16_supported()),
        gradient_checkpointing=(args.model == 'pythia'),
        logging_steps=1 if args.smoke else 25,
        report_to='none',
        dataloader_num_workers=0,
        optim='adamw_torch',
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=train_args,
        train_dataset=training_data, eval_dataset=validation_data,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )
    try:
        result = trainer.train()
        val = trainer.evaluate()
        trainer.save_model(str(out / 'final_model'))
        tokenizer.save_pretrained(str(out / 'final_model'))
        (out / 'validation_metrics.json').write_text(json.dumps(val, indent=2) + '\n')
        (out / 'training_metrics.json').write_text(
            json.dumps(result.metrics, indent=2) + '\n')
        print('SUCCESS. Validation metrics:', json.dumps(val, indent=2), flush=True)
        print('Test data NOT examined. Final model:', out / 'final_model', flush=True)
    finally:
        del trainer


if __name__ == '__main__':
    main()

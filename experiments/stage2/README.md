# FAIR-AIA Stage 2: Controlled fine-tuning + explainability

**Status:** Runnable research workflow prepared and syntax/synthetic dataset tests passed; NOT executed with the real PAN18 data or on the university GPU VM. The first GPU smoke test may require version-specific fixes.

This package creates a **new** experiment directory. It does not edit any previous scripts or paper files.

## Scientific plan

- Data: PAN18 English, first 20 tweets per author, joined and capped at 800 characters.
- Labels: female=0, male=1, following the dataset's original labels (not a general definition of gender).
- Seed: 2026; author-disjoint train/validation/test: 900/300/300 **per gender**.
- Models: `roberta-large`, `facebook/bart-large`, `EleutherAI/pythia-6.9b` base checkpoints, no MNLI adapters.
- Training: two-class sequence classification; RoBERTa/BART full fine-tuning; Pythia QLoRA-style 4-bit NF4 LoRA with a trainable classification head.
- Selection: validate every epoch; best validation AUC. NEVER use test labels to choose hyperparameters.
- Final evaluation: a single model-selected checkpoint, once on test, author-level AUC, male/female accuracies, signed gap, predicted-male fraction; stratified bootstrap + paired comparisons.
- Explainability: pilot LIME and SHAP on **validation** authors, then later predeclared test-author analysis. Word perturbations are not proofs of hidden-layer mechanisms.

## 0. Transfer and basic VM checks

Download `FAIR_AIA_Stage2.zip` from ChatGPT and transfer it to the VM. The example below assumes you put it in `~/pan18_project/`:

```bash
cd ~/pan18_project
unzip -n FAIR_AIA_Stage2.zip
cd FAIR_AIA_Stage2
pwd
nvidia-smi
python3 --version
python3 - <<'PY'
import importlib.metadata as m
for package in ('torch','transformers','accelerate','peft','bitsandbytes','lime','shap','scikit-learn'):
    try: print(package, m.version(package))
    except m.PackageNotFoundError: print(package, 'not installed')
PY
```

**Do not launch training on a GPU that someone else is using.** Choose an idle device. Run one training process per available GPU, initially only one.

## 1. Dependencies

Keep your existing PyTorch/CUDA installation if it works. A separate environment is advisable. On your VM, `python3` was previously 3.13 and PyTorch 2.5.1+cu121, but those may change. If using the existing packages via a venv:

```bash
cd ~/pan18_project/FAIR_AIA_Stage2
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install accelerate peft bitsandbytes lime shap
python -m pip check
python - <<'PY'
import torch, transformers, accelerate, peft, bitsandbytes, lime, shap, sklearn
print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
PY
```

**Important:** Package compatibility is not yet verified on your actual VM. If the installation tries to replace your existing PyTorch/CUDA build, STOP and inspect the proposed changes before proceeding. A clean project environment is preferable to modifying the shared base environment. Do not run `pip install --upgrade torch` blindly.

## 2. Unit test (CPU only)

```bash
python -m unittest discover -s tests -v
```

This creates tiny **synthetic** PAN-like author files in a temporary directory and verifies deterministic, balanced, non-overlapping splits and protection against overwrites.

## 3. Prepare the real PAN18 author splits

The dataset root was taken from your earlier scripts. This step reads the original files and writes new JSONL files into `data/`.

```bash
python scripts/prepare_data.py \
  --dataset-root ~/pan18/pan18-author-profiling-training-dataset-2018-02-27 \
  --out ./data --seed 2026
cat data/manifest.json
```

**Expected:** 3000 unique authors; train=1800 (900 female, 900 male); validation=600 (300 each); test=600 (300 each); `disjoint_author_ids: true`.

If the loader reports a missing label file or unexpected PAN layout, check `ls ~/pan18/pan18-author-profiling-training-dataset-2018-02-27/` and adapt the loader ONLY after looking at the actual files. Do not shuffle tweets or silently ignore missing authors.

The `data/` directory contains raw author text and is **private research data**. Do not push it to GitHub or include it in a public repository.

## 4. Choose an available GPU

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
nvidia-smi
```

For the examples below, replace `CUDA_VISIBLE_DEVICES=0` by `=1` if GPU 1 is available instead. Inside the process, a single visible GPU appears as `cuda:0`.

## 5. Smoke-test RoBERTa **before full training**

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py \
  --model roberta --data-dir ./data --out ./runs/roberta_smoke --smoke
```

This uses only 32 training authors, 16 validation authors, and four optimizer steps. It **never opens the test set**.

The goal is only to verify that loading, loss calculation, validation, and checkpoint saving work. The smoke metrics are **not publishable**.

If this fails, preserve the complete traceback; do not delete older experiments or rewrite original source files.

## 6. Full RoBERTa and BART runs

Run these separately, after the smoke test succeeds, and only on an idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py \
  --model roberta --data-dir ./data --out ./runs/roberta_full \
  --epochs 3 --lr 2e-5 --max-length 256

CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py \
  --model bart --data-dir ./data --out ./runs/bart_full \
  --epochs 3 --lr 2e-5 --max-length 256
```

These learning rates and epochs are **starting configurations**, not tuned results. Check validation AUC and loss after each epoch. If OOM, reduce `--batch-size` to 1, and increase `--grad-accum` to preserve the effective batch size. If needed, compare `--max-length 256` and `384` using validation data only.

Each successful run creates `final_model/`, `settings.json`, `validation_metrics.json`, and training history.

## 7. Pythia QLoRA-style training

After the smaller models are stable:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py \
  --model pythia --data-dir ./data --out ./runs/pythia_smoke --smoke
```

Then, if the smoke test works:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py \
  --model pythia --data-dir ./data --out ./runs/pythia_full \
  --epochs 3 --lr 2e-4 --batch-size 1 --grad-accum 8 --max-length 256
```

This loads Pythia in 4-bit NF4 precision, applies LoRA to GPT-NeoX attention's `query_key_value`, and also trains/saves the sequence classification head (`score`). The 4-bit/PEFT combination may need environment-specific fixes and has not yet been run on your L4 GPUs. Do not claim it was trained until the log and saved checkpoints confirm success.

## 8. Re-evaluate validation scores before accessing test

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model-dir ./runs/roberta_full/final_model --data-dir ./data \
  --split validation --out ./reports/roberta_val

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model-dir ./runs/bart_full/final_model --data-dir ./data \
  --split validation --out ./reports/bart_val

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model-dir ./runs/pythia_full/final_model --data-dir ./data \
  --split validation --out ./reports/pythia_val
```

Use validation results to lock the model/checkpoint and settings. Do NOT pick a model based on test performance.

## 9. Final held-out test, once the design is locked

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model-dir ./runs/roberta_full/final_model --data-dir ./data \
  --split test --out ./reports/roberta_test

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model-dir ./runs/bart_full/final_model --data-dir ./data \
  --split test --out ./reports/bart_test

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model-dir ./runs/pythia_full/final_model --data-dir ./data \
  --split test --out ./reports/pythia_test

python scripts/report.py \
  --pred roberta=./reports/roberta_test/predictions.jsonl \
         bart=./reports/bart_test/predictions.jsonl \
         pythia=./reports/pythia_test/predictions.jsonl \
  --bootstraps 2000 --seed 2026 --out ./reports/summary.json
```

The report calculates author-stratified bootstrap CIs and paired AUC-difference CIs using identical held-out author IDs. CIs do not prove model-family causation.

## 10. LIME and SHAP pilots on validation data

Start with RoBERTa and ONE author. These explainers can be slow because they query the model many times.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/explain.py \
  --model-dir ./runs/roberta_full/final_model \
  --data-dir ./data --split validation \
  --method lime --index 0 --n 1 --lime-samples 250 \
  --out ./explanations/roberta_lime_pilot

CUDA_VISIBLE_DEVICES=0 python scripts/explain.py \
  --model-dir ./runs/roberta_full/final_model \
  --data-dir ./data --split validation \
  --method shap --index 0 --n 1 --shap-evals 600 \
  --out ./explanations/roberta_shap_pilot
```

Both are local explanations for the model's **probability of the `male` label**. LIME produces a standalone HTML file and JSON weights; SHAP produces JSON word/token attributions and may generate HTML when the installed SHAP version supports it. For very long inputs, `--shap-evals` may need to be increased; do not silently shorten the explanation text and then treat it as an explanation of the original full-text prediction.

After pilots, select a prespecified, balanced set of correct and incorrect validation examples from each model. Compare explanation stability across random seeds and validate whether removing highlighted words measurably changes model outputs. Rebuild any gender-association vocabulary using **training authors only**; avoid contaminating the held-out test with old word-selection pools.

## Critical interpretation boundaries

- AUC describes how well predicted scores rank male-labeled authors relative to female-labeled authors. It is not accuracy or calibrated probability.
- Fine-tuning all three for a common task improves comparability but does **not** identify architecture as the sole cause of differences.
- The original Stage 1 model checkpoints and task setups differ from Stage 2. Comparisons must state that difference and should not treat the Stage 1 numbers as matched baselines.
- SHAP/LIME provide explanations of the *observable prediction function under masking/perturbation choices*, not proof of internal mechanisms.
- LIME/SHAP results should not be described as a completed experiment before they're actually executed and validated.
- This package is a **starting implementation** and not a validated finished research result. Preserve the exact environment, checkpoints, model revision, seeds, splits, logs, and output hashes for reproducibility.

## Troubleshooting checklist

- **PAN folder differs:** inspect `en.txt` and `en/` using `ls`; never guess labels.
- **CUDA OOM:** stop the failing process, check `nvidia-smi`, use a smaller batch and/or shorter token sequence; do not interrupt another person's process.
- **HF/PEFT version errors:** capture the error and `pip freeze`; avoid changing the shared base environment.
- **SHAP issue:** try LIME first; reproduce the exact error and check installed SHAP API before editing the code.
- **Poor AUC:** inspect class mapping, validation set, training loss, selected checkpoint, output distribution, and possible truncation; do not select settings using test performance.
- **Version mismatch:** `python -m pip check`, print `torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`, `shap`, `lime` versions.

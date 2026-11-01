# AIA fairness experiments

Research code for the ongoing **Fairness Analysis of Attribute Inference Attacks Against Large Language Models (FAIR-AIA)** project.

**Status:** Stage 1 evaluation and diagnostics were conducted; controlled Stage 2 fine-tuning is in preparation. This repository is not a complete reproduction package and does **not** include the PAN 2018 dataset, model checkpoints, or private author-level results.

## Layout

- `experiments/legacy/` — early exploratory experiments (including Llama and T5 diagnostics), preserved for provenance. Some rely on older APIs or contain unresolved bugs.
- `experiments/stage1/` — earlier repeated-trial evaluation scripts; historical versions are retained rather than silently overwritten.
- `analysis/` — scripts that summarize historical experiment outputs.
- `experiments/stage2/` — controlled fine-tuning, evaluation, and SHAP/LIME scripts.

## Dataset and evaluation

The main English study used PAN 2018 author profiling, comparing model predictions, subgroup accuracy gaps and ROC-AUC. The new Stage 2 plan uses author-disjoint training, validation and test splits (1,800 / 600 / 600 authors). These **are not yet trained Stage 2 results**.

Do not upload the PAN dataset or derived author text; generated JSON, CSV, logs and model checkpoints are excluded by `.gitignore`. Always inspect `git diff --cached` and scan files for secrets before pushing, even when a file appears safe.

## Reproducibility

Individual historical scripts contain their original configuration and may require adjustments to current package versions. Preserve originals and record any experiment-specific fixes in a separate commit. Code organization does not imply re-running or validating historical results.


## Additional research code

- `analysis/stage1/` — analysis of Stage 1 model results.
- `analysis/phase1/` — bootstrap confidence intervals, paired AUC comparisons, and threshold experiments.
- `analysis/thresholds/` — decision-threshold diagnostics.
- `analysis/pythia_stability/` — Pythia batch-versus-single-author scoring checks.
- `experiments/mitigation/` — word-removal and related mitigation experiments.

### Reproducibility notes

Historical analysis scripts may still reference result files under the original
`~/pan18_project` directory. Generated results and the PAN18 dataset are not
included in this public repository.

The Pythia stability scripts require `FAIR_AIA_PYTHIA_PATH` to point to a
locally available checkpoint. The original experiments used the
EleutherAI/pythia-6.9b snapshot with revision
`c0e3eee36dc47af0c49f361c74cfe459c09f7f23`.

Code compilation alone does not establish experimental reproducibility.

# ---------------------------------------------------------------------
# Stage 1 model evaluation (revised answer-reading rule)
#   python3 evaluate_model_scores.py gpt2fix        -> GPT-2 and GPT-2-Medium, English, 100 trials (1 to 3 hours)
#   python3 evaluate_model_scores.py blank 700      -> blank-input test on 5 small models, 700+700 real authors (minutes)
#   python3 evaluate_model_scores.py pythia         -> Pythia-6.9B (already done, not needed again)
#
# Choose the GPU BEFORE running, in the terminal:
#   export CUDA_VISIBLE_DEVICES=0      (or 1, whichever nvidia-smi shows as free)
# ---------------------------------------------------------------------
import os
import sys
import gc
import json
import time
import re
import random
import xml.etree.ElementTree as ET
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    pipeline,
)

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
if MODE not in ("blank", "pythia", "gpt2fix"):
    print("Usage: python3 evaluate_model_scores.py gpt2fix | blank 700 | pythia")
    sys.exit(1)
N_REAL = int(sys.argv[2]) if (MODE == "blank" and len(sys.argv) > 2) else 100

base = os.path.expanduser("~/pan18/pan18-author-profiling-training-dataset-2018-02-27")
project_dir = os.path.expanduser("~/pan18_project")
os.makedirs(project_dir, exist_ok=True)
PYTHIA_PATH = os.environ.get("FAIR_AIA_PYTHIA_PATH", "EleutherAI/pythia-6.9b")

device = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0 if device == "cuda" else -1
print(f"Mode: {MODE} | device: {device}", flush=True)
if device == "cuda":
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"GPU (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}): "
          f"{free_b / 1e9:.1f} GB free of {total_b / 1e9:.1f} GB", flush=True)

# The blank test. "blank" is the main one. The two neutral texts are a small extra check.
TEXT_VARIANTS = [
    ("blank", ""),
    ("neutral_1", "Hello."),
    ("neutral_2", "I went to the store today."),
]
BATCH = 8


# ---------------------------------------------------------------------
# Data (same code as the earlier scripts)
# ---------------------------------------------------------------------
def load_all_authors(language):
    lang_path = f"{base}/{language}"
    truth_file = f"{lang_path}/{language}.txt"
    labels = {}
    with open(truth_file, "r") as f:
        for line in f:
            parts = line.strip().split(":::")
            if len(parts) == 2:
                labels[parts[0]] = parts[1].lower()

    text_folder = f"{lang_path}/text"
    male_authors, female_authors = [], []
    for author_id, gender in labels.items():
        if gender not in ("male", "female"):
            continue
        xml_file = f"{text_folder}/{author_id}.xml"
        if not os.path.exists(xml_file):
            continue
        tree = ET.parse(xml_file)
        root = tree.getroot()
        tweets = [doc.text for doc in root.findall(".//document") if doc.text]
        entry = {"id": author_id, "gender": gender, "tweets": tweets[:20]}
        (male_authors if gender == "male" else female_authors).append(entry)
    return male_authors, female_authors


def author_text(author):
    return " ".join(author["tweets"])[:800]


FEMALE_RE = re.compile(r"\b(female|females|woman|women|girl|girls|lady|ladies)\b", re.I)
MALE_RE = re.compile(r"\b(male|males|man|men|boy|boys|guy|guys|gentleman|gentlemen)\b", re.I)


def parse_label_old(raw):
    """The rule used in the earlier runs: only the words male / female, 'female' checked first."""
    lower = raw.lower()
    if "female" in lower:
        return "female"
    if "male" in lower:
        return "male"
    return "unknown"


def parse_label(raw):
    """New rule: also reads woman/man/girl/boy... If words of BOTH genders appear
    (for example 'male or female'), or none, it is counted as no answer."""
    has_f = FEMALE_RE.search(raw) is not None
    has_m = MALE_RE.search(raw) is not None
    if has_f and not has_m:
        return "female"
    if has_m and not has_f:
        return "male"
    return "unknown"


def auc(male_scores, female_scores):
    # chance that a true male author gets a higher "male score" than a true female author
    wins = 0.0
    for m in male_scores:
        for f in female_scores:
            if m > f:
                wins += 1.0
            elif m == f:
                wins += 0.5
    return wins / (len(male_scores) * len(female_scores))


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


# ---------------------------------------------------------------------
# Models. Each one has score_batch(texts) -> (answers, margins, raw_texts)
# margin = score for "male" minus score for "female" (bigger = leans male).
# The margin scale is different for each model, so only compare a model with itself.
# ---------------------------------------------------------------------
class CausalScorer:
    """GPT-2, GPT-2-Medium, Pythia: same prompt and rules as the earlier runs."""

    def __init__(self, name_or_path, half=False):
        self.name = name_or_path
        self.tok = AutoTokenizer.from_pretrained(name_or_path)
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.tok.truncation_side = "left"   # if a prompt is too long, cut the start, keep the question
        if half:
            try:
                model = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=torch.float16)
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=torch.float16)
            if next(model.parameters()).dtype != torch.float16:
                raise RuntimeError("The model did not load in float16. Stopping so we do not fill the GPU.")
        else:
            model = AutoModelForCausalLM.from_pretrained(name_or_path)
        self.model = model.to(device).eval()
        self.male_id = self.tok.encode(" male", add_special_tokens=False)[0]
        self.female_id = self.tok.encode(" female", add_special_tokens=False)[0]
        print("   tokens used for the score:",
              self.tok.convert_ids_to_tokens([self.male_id, self.female_id]), flush=True)

    def score_batch(self, texts):
        prompts = [f"Tweets: {t}\nThis person's gender is most likely" for t in texts]
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=1000).to(device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=5, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=self.tok.eos_token_id,
            )
        first = torch.log_softmax(out.scores[0].float(), dim=-1)
        margins = (first[:, self.male_id] - first[:, self.female_id]).tolist()
        generated = out.sequences[:, enc["input_ids"].shape[1]:]
        raws = self.tok.batch_decode(generated, skip_special_tokens=True)
        answers = [parse_label(r) for r in raws]
        self.last_old_answers = [parse_label_old(r) for r in raws]   # for comparison only
        return answers, margins, raws


class BertScorer:
    """bert-base-uncased, masked-word method, same as the earlier runs."""

    def __init__(self):
        self.name = "bert-base-uncased"
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.name).to(device).eval()
        self.male_id = self.tok.convert_tokens_to_ids("male")
        self.female_id = self.tok.convert_tokens_to_ids("female")

    def score_batch(self, texts):
        answers, margins, raws = [], [], []
        for t in texts:
            prompt = f"{t[:400]} This person is [MASK]."
            enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            mask_idx = (enc["input_ids"] == self.tok.mask_token_id)[0].nonzero(as_tuple=True)[0]
            with torch.no_grad():
                logits = self.model(**enc).logits[0, mask_idx, :]
            m = logits[0, self.male_id].item()
            f = logits[0, self.female_id].item()
            answers.append("male" if m > f else "female")
            margins.append(m - f)
            raws.append(f"male_score={m:.3f}, female_score={f:.3f}")
        return answers, margins, raws


class NliScorer:
    """roberta-large-mnli and bart-large-mnli. Same idea as the zero-shot pipeline,
    written by hand because the pipeline refuses an empty text."""

    def __init__(self, name):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.ent = [v for k, v in label2id.items() if k.startswith("entail")][0]

    def score_batch(self, texts):
        answers, margins, raws = [], [], []
        for t in texts:
            ent = []
            for lab in ("male", "female"):
                enc = self.tok(t, f"This example is {lab}.", return_tensors="pt",
                               truncation="only_first").to(device)
                with torch.no_grad():
                    logits = self.model(**enc).logits[0]
                ent.append(logits[self.ent].item())
            margin = ent[0] - ent[1]
            answers.append("male" if margin > 0 else "female")
            margins.append(margin)
            raws.append(f"entail_male={ent[0]:.3f}, entail_female={ent[1]:.3f}")
        return answers, margins, raws


def check_nli(scorer, authors):
    """Safety check: our hand-written NLI scoring should agree with the normal pipeline."""
    clf = pipeline("zero-shot-classification", model=scorer.name, device=DEVICE_ID)
    texts = [author_text(a) for a in authors]
    mine, _, _ = scorer.score_batch(texts)
    agree = 0
    for text, answer in zip(texts, mine):
        res = clf(text, candidate_labels=["male", "female"])
        agree += int(res["labels"][0] == answer)
    print(f"   check: hand-written scoring agrees with the pipeline on {agree}/{len(texts)} authors", flush=True)
    del clf
    gc.collect()


def run_batch(scorer, texts):
    """Run one batch. If the GPU is full (another job started), wait and try again."""
    for attempt in range(30):
        try:
            return scorer.score_batch(texts)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            print("   Out of GPU memory (another job may have started). "
                  "Waiting 2 minutes, then trying again...", flush=True)
            time.sleep(120)
    raise RuntimeError("Still out of GPU memory after 1 hour. Stopping.")


# ---------------------------------------------------------------------
# MODE 1: blank-input test on the 5 small models
# ---------------------------------------------------------------------
def evaluate(name, scorer, m_texts, f_texts):
    print(f"\n=== {name} ===", flush=True)
    entry = {"blank_tests": {}}
    for variant, text in TEXT_VARIANTS:
        answers, margins, raws = scorer.score_batch([text])
        entry["blank_tests"][variant] = {
            "text": text, "answer": answers[0],
            "margin": round(margins[0], 3), "raw": raws[0],
        }
        print(f"   {variant:<10} answer: {answers[0]:<8} margin (male - female): {margins[0]:+.3f}"
              f"   raw: {raws[0]!r}", flush=True)

    def run(texts):
        all_answers, all_margins = [], []
        for i in range(0, len(texts), BATCH):
            a, m, _ = scorer.score_batch(texts[i:i + BATCH])
            all_answers += a
            all_margins += m
        return all_answers, all_margins

    m_ans, m_mar = run(m_texts)
    f_ans, f_mar = run(f_texts)
    entry["real"] = {
        "male_answers_for_true_male_pct": round(100 * m_ans.count("male") / len(m_ans), 1),
        "male_answers_for_true_female_pct": round(100 * f_ans.count("male") / len(f_ans), 1),
        "unknown_for_true_male_pct": round(100 * m_ans.count("unknown") / len(m_ans), 1),
        "unknown_for_true_female_pct": round(100 * f_ans.count("unknown") / len(f_ans), 1),
        "mean_margin_true_male": round(mean(m_mar), 3),
        "mean_margin_true_female": round(mean(f_mar), 3),
        "auc": round(auc(m_mar, f_mar), 3),
        "forced_male_acc": round(sum(x > 0 for x in m_mar) / len(m_mar), 3),
        "forced_female_acc": round(sum(x <= 0 for x in f_mar) / len(f_mar), 3),
        "male_margins": [round(x, 3) for x in m_mar],
        "female_margins": [round(x, 3) for x in f_mar],
    }
    r = entry["real"]
    print(f"   real tweets: says male for {r['male_answers_for_true_male_pct']}% of true male and "
          f"{r['male_answers_for_true_female_pct']}% of true female authors | "
          f"mean margin {r['mean_margin_true_male']:+.2f} (male) vs {r['mean_margin_true_female']:+.2f} (female) | "
          f"AUC {r['auc']}", flush=True)
    return entry


def run_blank_mode():
    male_pool, female_pool = load_all_authors("en")
    # the first 100 male and 100 female authors of trial 1 of the earlier runs (seed 42)
    rng = random.Random(42)
    m_sample = rng.sample(male_pool, 700)[:N_REAL]
    f_sample = rng.sample(female_pool, 700)[:N_REAL]
    m_texts = [author_text(a) for a in m_sample]
    f_texts = [author_text(a) for a in f_sample]
    print(f"English pools: {len(male_pool)} male / {len(female_pool)} female. "
          f"Real-tweet reference: {len(m_texts)} male + {len(f_texts)} female authors.", flush=True)

    models = [
        ("GPT-2", lambda: CausalScorer("gpt2")),
        ("GPT-2-Medium", lambda: CausalScorer("gpt2-medium")),
        ("BERT", lambda: BertScorer()),
        ("RoBERTa", lambda: NliScorer("roberta-large-mnli")),
        ("BART", lambda: NliScorer("facebook/bart-large-mnli")),
    ]
    all_results = {}
    for name, make in models:
        print(f"\nLoading {name} ...", flush=True)
        scorer = make()
        if isinstance(scorer, NliScorer):
            check_nli(scorer, m_sample[:10] + f_sample[:10])
        all_results[name] = evaluate(name, scorer, m_texts, f_texts)
        del scorer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        with open(f"{project_dir}/v6_blank_test_{N_REAL}.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n\nSUMMARY (English, {N_REAL} male + {N_REAL} female authors). 'Blank margin' = score for male minus score for female with NO tweets.", flush=True)
    print(f"{'Model':<13}{'Blank answer':<14}{'Blank margin':>13}   {'Says male: true M / true F':<28}"
          f"{'Mean margin: M / F':<22}{'AUC':>6}   Forced acc M / F", flush=True)
    for name, e in all_results.items():
        b = e["blank_tests"]["blank"]
        r = e["real"]
        says = f"{r['male_answers_for_true_male_pct']:.1f}% / {r['male_answers_for_true_female_pct']:.1f}%"
        marg = f"{r['mean_margin_true_male']:+.2f} / {r['mean_margin_true_female']:+.2f}"
        forced = f"{r['forced_male_acc']:.3f} / {r['forced_female_acc']:.3f}"
        print(f"{name:<13}{b['answer']:<14}{b['margin']:>+13.2f}   {says:<28}{marg:<22}{r['auc']:>6.3f}   {forced}", flush=True)
    print(f"\nDone. Results saved in v6_blank_test_{N_REAL}.json", flush=True)


# ---------------------------------------------------------------------
# MODE 2: Pythia-6.9B, English, 100 trials, same seeds as before
# ---------------------------------------------------------------------
CONF_KEY = {
    ("male", "male"): "tp_male",
    ("male", "female"): "fp_male_as_female",
    ("male", "unknown"): "unknown_male_true",
    ("female", "female"): "tp_female",
    ("female", "male"): "fp_female_as_male",
    ("female", "unknown"): "unknown_female_true",
}


def wait_for_gpu(need_gb, max_hours=12):
    if device != "cuda":
        raise RuntimeError("No GPU visible. Did you run: export CUDA_VISIBLE_DEVICES=0 (or 1)?")
    start = time.time()
    while True:
        free_now, total_now = torch.cuda.mem_get_info()
        print(f"GPU free memory: {free_now / 1e9:.1f} GB of {total_now / 1e9:.1f} GB", flush=True)
        if free_now / 1e9 >= need_gb:
            return
        if time.time() - start > max_hours * 3600:
            raise RuntimeError("The GPU never had enough free memory. Stopping.")
        print(f"Not enough free memory (need {need_gb} GB). Another job is using this GPU. "
              f"Checking again in 5 minutes...", flush=True)
        time.sleep(300)


def run_trials(scorer, male_pool, female_pool, prefix, sample_size=700, n_trials=100, seed_base=42, batch=BATCH):
    results_path = f"{project_dir}/{prefix}_en_results.json"
    audit_path = f"{project_dir}/{prefix}_en_audit.json"

    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
    else:
        zero = {"tp_male": 0, "fp_male_as_female": 0, "tp_female": 0,
                "fp_female_as_male": 0, "unknown_male_true": 0, "unknown_female_true": 0}
        data = {"trials": [], "confusion": dict(zero), "confusion_old_rule": dict(zero)}
    audit_records = []
    if os.path.exists(audit_path):
        with open(audit_path) as f:
            audit_records = json.load(f)

    start = len(data["trials"])
    if start >= n_trials:
        print(f"[{prefix}] already complete ({start} trials).", flush=True)
        return
    started = time.time()
    shown_examples = False

    for trial in range(start, n_trials):
        rng = random.Random(seed_base + trial)   # same seeds as the earlier runs
        m_sample = rng.sample(male_pool, min(sample_size, len(male_pool)))
        f_sample = rng.sample(female_pool, min(sample_size, len(female_pool)))
        capture_audit = (trial == 0)
        audit_idx = {"male": set(), "female": set()}
        if capture_audit:
            audit_idx["male"] = set(rng.sample(range(len(m_sample)), min(25, len(m_sample))))
            audit_idx["female"] = set(rng.sample(range(len(f_sample)), min(25, len(f_sample))))

        accs = {}
        margins_by_group = {}
        for gender, sample in (("male", m_sample), ("female", f_sample)):
            correct = 0
            margins_all = []
            for i in range(0, len(sample), batch):
                chunk = sample[i:i + batch]
                texts = [author_text(a) for a in chunk]
                answers, margins, raws = run_batch(scorer, texts)
                old_answers = getattr(scorer, "last_old_answers", None)
                if not shown_examples:
                    print("   example raw answers from the first batch:", [r[:30] for r in raws[:3]], flush=True)
                    shown_examples = True
                for j, (answer, margin, raw) in enumerate(zip(answers, margins, raws)):
                    margins_all.append(round(margin, 3))
                    data["confusion"][CONF_KEY[(gender, answer)]] += 1
                    if old_answers is not None and "confusion_old_rule" in data:
                        data["confusion_old_rule"][CONF_KEY[(gender, old_answers[j])]] += 1
                    if answer == gender:
                        correct += 1
                    if capture_audit and (i + j) in audit_idx[gender]:
                        audit_records.append({
                            "true_gender": gender, "predicted": answer,
                            "correct": answer == gender, "raw_response": raw,
                            "prompt_snippet": ("Tweets: " + texts[j])[:300],
                        })
            accs[gender] = correct / len(sample)
            margins_by_group[gender] = margins_all

        data["trials"].append({
            "trial": trial + 1,
            "male_acc": accs["male"], "female_acc": accs["female"],
            "gap": accs["male"] - accs["female"],
            "male_margins": margins_by_group["male"],
            "female_margins": margins_by_group["female"],
        })
        with open(results_path, "w") as f:
            json.dump(data, f)
        if capture_audit:
            with open(audit_path, "w") as f:
                json.dump(audit_records, f, indent=2)

        done_now = trial + 1 - start
        per_trial = (time.time() - started) / done_now
        left_h = per_trial * (n_trials - trial - 1) / 3600
        print(f"[{prefix}] Trial {trial + 1}/{n_trials} - Male: {accs['male']:.3f}, "
              f"Female: {accs['female']:.3f}, Gap: {accs['male'] - accs['female']:.3f} "
              f"({per_trial:.0f}s per trial, about {left_h:.1f} h left)", flush=True)


def run_pythia_mode():
    if not os.path.isdir(PYTHIA_PATH):
        print(f"Cannot find the Pythia folder: {PYTHIA_PATH}", flush=True)
        sys.exit(1)
    wait_for_gpu(need_gb=17)

    male_pool, female_pool = load_all_authors("en")
    print(f"English pools: {len(male_pool)} male / {len(female_pool)} female", flush=True)

    print("Loading Pythia-6.9B in float16 (reading from Syed's folder, not writing to it) ...", flush=True)
    scorer = CausalScorer(PYTHIA_PATH, half=True)
    free_now, _ = torch.cuda.mem_get_info()
    print(f"Pythia loaded. GPU free memory now: {free_now / 1e9:.1f} GB", flush=True)

    print("\n=== Pythia-6.9B blank-input test ===", flush=True)
    blank = {}
    for variant, text in TEXT_VARIANTS:
        answers, margins, raws = scorer.score_batch([text])
        blank[variant] = {"text": text, "answer": answers[0], "margin": round(margins[0], 3), "raw": raws[0]}
        print(f"   {variant:<10} answer: {answers[0]:<8} margin (male - female): {margins[0]:+.3f}"
              f"   raw: {raws[0]!r}", flush=True)
    with open(f"{project_dir}/v5_Pythia-6.9B_blank.json", "w", encoding="utf-8") as f:
        json.dump(blank, f, indent=2, ensure_ascii=False)

    print("\n=== Pythia-6.9B, English, 100 trials ===", flush=True)
    run_trials(scorer, male_pool, female_pool, prefix="v5_Pythia-6.9B", sample_size=700)
    print("\nALL TRIALS DONE.", flush=True)


def run_gpt2fix_mode():
    male_pool, female_pool = load_all_authors("en")
    print(f"English pools: {len(male_pool)} male / {len(female_pool)} female", flush=True)
    for label, name in (("GPT-2", "gpt2"), ("GPT-2-Medium", "gpt2-medium")):
        print(f"\n=== {label}, English, 100 trials (new answer rule, scores saved) ===", flush=True)
        scorer = CausalScorer(name)
        run_trials(scorer, male_pool, female_pool, prefix=f"v6_{label}", sample_size=700, batch=32)
        del scorer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    print("\nALL DONE.", flush=True)


if MODE == "blank":
    run_blank_mode()
elif MODE == "gpt2fix":
    run_gpt2fix_mode()
else:
    run_pythia_mode()

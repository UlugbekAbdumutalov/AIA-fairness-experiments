"""run_mitigation.py - hide gender words, re-run one model, save results.
Usage:  python3 -u run_mitigation.py --model gpt2 [--trials 5] [--per-group 300] [--smoke]"""
import os, sys, csv, time, json, argparse
import numpy as np
from common import (load_authors, rank_gender_words, matched_random, make_hider,
                    evaluate, words)
from scorers import get_scorer, max_chars, MODELS

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, choices=list(MODELS))
p.add_argument("--trials", type=int, default=5)
p.add_argument("--per-group", type=int, default=300, help="authors per gender in each trial")
p.add_argument("--choose", type=int, default=500, help="authors per gender used ONLY to find the words")
p.add_argument("--levels", default="10,50,200,1000", help="how many words to hide")
p.add_argument("--smoke", action="store_true", help="tiny quick test")
p.add_argument("--data-root", default=None)
a = p.parse_args()
if a.smoke:
    a.trials, a.per_group, a.levels = 1, 20, "10,50"
levels = [int(x) for x in a.levels.split(",")]

# ---- 1. data
kw = {"root": a.data_root} if a.data_root else {}
ids, texts, genders = load_authors(**kw)
genders = np.array(genders)
print(f"authors: {len(ids)}  male={int((genders=='male').sum())}  female={int((genders=='female').sum())}", flush=True)

# ---- 2. split: 'choose' authors (find words) and 'test' pool (never used to find words)
rs = np.random.RandomState(42)
choose_idx, pool = [], {}
for g in ("male", "female"):
    idx = np.where(genders == g)[0]
    rs.shuffle(idx)
    choose_idx += idx[:a.choose].tolist()
    pool[g] = idx[a.choose:]
    if len(pool[g]) < a.per_group:
        sys.exit(f"Not enough {g} authors in the test pool ({len(pool[g])}). Lower --choose or --per-group.")
rows = rank_gender_words([texts[i] for i in choose_idx], [genders[i] for i in choose_idx])
print(f"vocabulary with >=5 authors: {len(rows)} words. Top 15 gender words:", flush=True)
print("  ", [(w, round(z, 1)) for w, z, _ in rows[:15]], flush=True)
os.makedirs("results", exist_ok=True)
json.dump(rows[:max(levels)], open("results/gender_words.json", "w"))

# ---- 3. model
scorer = get_scorer(a.model)
mc = max_chars(a.model)

# ---- 4. output file (resume-safe)
out_path = f"results/mitigation_{a.model}.csv"
done = set()
fields = ["model", "trial", "condition", "k", "removed_pct", "n_authors", "auc",
          "male_acc_def", "female_acc_def", "gap_def", "avg_acc_def",
          "male_acc_med", "female_acc_med", "gap_med", "avg_acc_med"]
if os.path.exists(out_path):
    for r in csv.DictReader(open(out_path)):
        done.add((int(r["trial"]), r["condition"], int(r["k"])))
else:
    with open(out_path, "w", newline="") as fh:
        csv.DictWriter(fh, fields).writeheader()

conditions = [("none", 0)] + [(c, k) for k in levels for c in ("gender", "random")]

# ---- 5. trials
t0 = time.time()
for trial in range(a.trials):
    rng = np.random.RandomState(42 + trial)                      # same seeds idea as the main paper
    sel = np.concatenate([rng.choice(pool["male"], a.per_group, replace=False),
                          rng.choice(pool["female"], a.per_group, replace=False)])
    is_male = np.array([genders[i] == "male" for i in sel])
    base = [texts[i][:mc] for i in sel]                          # cut to the same length as the main test
    rng_ctrl = np.random.RandomState(1000 + trial)
    for cond, k in conditions:
        if (trial, cond, k) in done:
            continue
        top = rows[:k]
        if cond == "none":
            wl = []
        elif cond == "gender":
            wl = [r[0] for r in top]
        else:
            wl = matched_random(top, rows, rng_ctrl)
        hide = make_hider(wl)
        new = [hide(t) for t in base]
        n_before = sum(len(words(t)) for t in base)
        n_after = sum(len(words(t)) for t in new)
        removed = 100.0 * (1 - n_after / max(n_before, 1))
        sc = scorer(new)
        m = evaluate(np.asarray(sc, dtype=float), is_male)
        row = {"model": a.model, "trial": trial, "condition": cond, "k": k,
               "removed_pct": round(removed, 2), "n_authors": len(sel)}
        row.update({kk: round(v, 4) for kk, v in m.items()})
        with open(out_path, "a", newline="") as fh:
            csv.DictWriter(fh, fields).writerow(row)           # checkpoint after every step
        print(f"[{a.model}] trial {trial} {cond:6s} k={k:<5d} removed={removed:5.1f}%  "
              f"AUC={m['auc']:.3f}  avg_acc(median)={m['avg_acc_med']:.3f}  "
              f"gap(default)={m['gap_def']:+.3f}   ({time.time()-t0:.0f}s)", flush=True)
print("finished", a.model, flush=True)

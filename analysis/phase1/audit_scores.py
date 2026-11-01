import json
import math
from pathlib import Path
from bisect import bisect_left, bisect_right
from statistics import median

ROOT = Path.home() / "pan18_project"

def load(name):
    path = ROOT / name
    if not path.exists():
        print(f"MISSING: {name}")
        return None
    with path.open() as f:
        return json.load(f)

def auc(ms, fs):
    fs = sorted(fs)
    wins = 0.0
    for x in ms:
        lo = bisect_left(fs, x)
        hi = bisect_right(fs, x)
        wins += lo + (hi - lo) / 2
    return wins / (len(ms) * len(fs))

def evaluate(ms, fs, cutoff):
    m = sum(x > cutoff for x in ms) / len(ms)
    f = sum(x <= cutoff for x in fs) / len(fs)
    return m, f, m - f, (m + f) / 2

def report(name, ms, fs, trials=None, author_ids=False):
    if not ms or not fs:
        print(f"{name}: missing scores")
        return
    if not all(isinstance(x, (int, float)) and math.isfinite(x)
               for x in ms + fs):
        print(f"{name}: invalid/non-finite scores found")
        return

    cut = median(ms + fs)
    a = auc(ms, fs)
    default = evaluate(ms, fs, 0.0)
    med = evaluate(ms, fs, cut)
    ties = sum(x == cut for x in ms + fs)
    reversed_auc = 1.0 - a

    print(f"\n{'=' * 64}")
    print(f"MODEL: {name}")
    print(f"Male score entries:   {len(ms)}")
    print(f"Female score entries: {len(fs)}")
    print(f"Number of trials:     {trials if trials is not None else 'single saved sample'}")
    print(f"Author IDs found:     {author_ids}")
    print(f"AUC:                  {a:.6f}")
    print(f"AUC reversed:         {reversed_auc:.6f}")
    print(f"Default cutoff:       0.000000")
    print(f"Default male acc:     {default[0]:.6f}")
    print(f"Default female acc:   {default[1]:.6f}")
    print(f"Default gap:          {default[2]:+.6f}")
    print(f"Median cutoff:        {cut:.6f}")
    print(f"Median ties:          {ties}")
    print(f"Median male acc:      {med[0]:.6f}")
    print(f"Median female acc:    {med[1]:.6f}")
    print(f"Median gap:           {med[2]:+.6f}")
    print(f"Median average acc:   {med[3]:.6f}")
    if len(ms) == len(fs):
        predicted_male_fraction = sum(x > cut for x in ms + fs) / (len(ms) + len(fs))
        print(f"Median predicted male fraction: {predicted_male_fraction:.6f}")
        print(f"Formula 2*q-1:        {2 * predicted_male_fraction - 1:+.6f}")
        print(f"Matches median gap:   {abs(med[2] - (2 * predicted_male_fraction - 1)) < 1e-9}")

print("PHASE 1: READ-ONLY CUTOFF AUDIT")
print("No models or GPUs are used. Original data are not changed.")

for name in ("GPT-2", "GPT-2-Medium", "Pythia-6.9B"):
    prefix = "v5" if name == "Pythia-6.9B" else "v6"
    d = load(f"{prefix}_{name}_en_results.json")
    if not d:
        continue

    trials = d.get("trials", [])
    ms = [x for t in trials for x in t.get("male_margins", [])]
    fs = [x for t in trials for x in t.get("female_margins", [])]
    has_ids = any(any("id" in str(k).lower() for k in t.keys())
                  for t in trials)
    report(name, ms, fs, len(trials), has_ids)

    if trials:
        gaps = []
        for t in trials:
            tm, tf = t.get("male_margins", []), t.get("female_margins", [])
            if tm and tf and len(tm) == len(tf):
                threshold = median(tm + tf)
                gaps.append(evaluate(tm, tf, threshold)[2])
        if gaps:
            print(f"Mean within-trial median gap: {sum(gaps)/len(gaps):+.6f}")

blank = load("v6_blank_test_700.json")
if blank:
    for name in ("BERT", "RoBERTa", "BART"):
        if name not in blank:
            continue
        real = blank[name].get("real", {})
        ms = real.get("male_margins", [])
        fs = real.get("female_margins", [])
        report(name, ms, fs)

print("\n" + "=" * 64)
print("AUDIT FINISHED")
print("IMPORTANT: repeated score entries may come from the same author.")
print("A new author-disjoint test requires author IDs or fresh scoring.")

import os
import json
from bisect import bisect_right

project_dir = os.path.expanduser("~/pan18_project")


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def auc(male_scores, female_scores):
    fs = sorted(female_scores)
    from bisect import bisect_left
    wins = 0.0
    for x in male_scores:
        lo = bisect_left(fs, x)
        hi = bisect_right(fs, x)
        wins += lo + 0.5 * (hi - lo)
    return wins / (len(male_scores) * len(female_scores))


def accs(t, ms, fs):
    """ms, fs = sorted scores of true male / true female authors.
    The model says 'male' when its score is above the cut-off t."""
    male_acc = (len(ms) - bisect_right(ms, t)) / len(ms)
    female_acc = bisect_right(fs, t) / len(fs)
    return male_acc, female_acc


def median(sorted_xs):
    n = len(sorted_xs)
    return (sorted_xs[n // 2] + sorted_xs[(n - 1) // 2]) / 2


# ---------------------------------------------------------------
# Collect the saved scores (male score minus female score) for every model
# ---------------------------------------------------------------
models = {}   # name -> (male scores, female scores, blank score)
blank700 = load(f"{project_dir}/v6_blank_test_700.json") or {}

for name in ("GPT-2", "GPT-2-Medium"):
    d = load(f"{project_dir}/v6_{name}_en_results.json")
    if d is not None:
        ms = [x for t in d["trials"] for x in t["male_margins"]]
        fs = [x for t in d["trials"] for x in t["female_margins"]]
        blank = blank700.get(name, {}).get("blank_tests", {}).get("blank", {}).get("margin")
        models[name] = (ms, fs, blank)

py = load(f"{project_dir}/v5_Pythia-6.9B_en_results.json")
if py is not None:
    ms = [x for t in py["trials"] for x in t["male_margins"]]
    fs = [x for t in py["trials"] for x in t["female_margins"]]
    pb = load(f"{project_dir}/v5_Pythia-6.9B_blank.json")
    models["Pythia-6.9B"] = (ms, fs, pb["blank"]["margin"] if pb else None)

for name in ("BERT", "RoBERTa", "BART"):
    if name in blank700:
        r = blank700[name]["real"]
        models[name] = (r["male_margins"], r["female_margins"], blank700[name]["blank_tests"]["blank"]["margin"])

# ---------------------------------------------------------------
# Try four cut-offs for each model
#   1. default      : cut-off 0 (what we used so far)
#   2. blank score  : cut-off = the score the model gives with NO tweets (needs no labels)
#   3. median       : cut-off so the model says 'male' for half of all authors (needs no labels)
#   4. equal-error  : cut-off that makes male accuracy = female accuracy (uses the labels, best case)
# ---------------------------------------------------------------
print("What happens if we only move the cut-off? (English)")
print("Gap = male accuracy - female accuracy.  Avg = (male accuracy + female accuracy) / 2, the average attack strength.\n")
header = f"{'Model':<14}{'AUC':>6} | {'Default':^17} | {'Blank score':^17} | {'Median':^17} | {'Equal-error':^17}"
print(header)
print(f"{'':<14}{'':>6} | {'gap':>8}{'avg':>9} | {'gap':>8}{'avg':>9} | {'gap':>8}{'avg':>9} | {'gap':>8}{'avg':>9}")

for name in ("GPT-2", "GPT-2-Medium", "Pythia-6.9B", "BERT", "RoBERTa", "BART"):
    if name not in models:
        continue
    male, female, blank_score = models[name]
    ms, fs = sorted(male), sorted(female)
    allx = sorted(ms + fs)
    a = auc(male, female) if len(male) * len(female) <= 5_000_000 else None
    if a is None:   # big lists (70,000 x 70,000): use the sorted-list shortcut
        wins = 0.0
        from bisect import bisect_left
        for x in ms:
            lo = bisect_left(fs, x)
            hi = bisect_right(fs, x)
            wins += lo + 0.5 * (hi - lo)
        a = wins / (len(ms) * len(fs))

    cutoffs = {"default": 0.0}
    if blank_score is not None:
        cutoffs["blank"] = blank_score
    cutoffs["median"] = median(allx)
    best_t, best_diff = 0.0, 9.0
    for t in sorted(set(allx)):
        m_acc, f_acc = accs(t, ms, fs)
        if abs(m_acc - f_acc) < best_diff:
            best_diff, best_t = abs(m_acc - f_acc), t
    cutoffs["equal"] = best_t

    cells = []
    for key in ("default", "blank", "median", "equal"):
        if key in cutoffs:
            m_acc, f_acc = accs(cutoffs[key], ms, fs)
            cells.append(f"{m_acc - f_acc:>+8.3f}{(m_acc + f_acc) / 2:>9.3f}")
        else:
            cells.append(f"{'n/a':>8}{'n/a':>9}")
    print(f"{name:<14}{a:>6.3f} | " + " | ".join(cells))

print("\nHow to read it:")
print("  - A gap close to 0 with the same 'avg' means the unfairness was only the cut-off, not the model reading the text.")
print("  - If AUC is below 0.5 the score points the wrong way, so moving the cut-off cannot help.")
print("  - 'Blank score' and 'Median' use no labels. 'Equal-error' uses the labels, so it is the best case.")

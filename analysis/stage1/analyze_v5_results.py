import os
import json
import random
import statistics as stats
from bisect import bisect_left, bisect_right

project_dir = os.path.expanduser("~/pan18_project")
PYTHIA_RESULTS = f"{project_dir}/v5_Pythia-6.9B_en_results.json"
PYTHIA_BLANK = f"{project_dir}/v5_Pythia-6.9B_blank.json"
SMALL_BLANK = f"{project_dir}/v5_blank_test_results.json"


def auc(male_scores, female_scores):
    # chance that a true male author gets a higher "male score" than a true female author
    fs = sorted(female_scores)
    wins = 0.0
    for x in male_scores:
        lo = bisect_left(fs, x)
        hi = bisect_right(fs, x)
        wins += lo + 0.5 * (hi - lo)
    return wins / (len(male_scores) * len(female_scores))


def mean(xs):
    return sum(xs) / len(xs)


# ---------------------------------------------------------------
# 1. Pythia-6.9B, English, 100 trials
# ---------------------------------------------------------------
d = json.load(open(PYTHIA_RESULTS))
trials = d["trials"]
m_acc = [t["male_acc"] for t in trials]
f_acc = [t["female_acc"] for t in trials]
gaps = [t["gap"] for t in trials]

random.seed(0)
boot = sorted(mean([random.choice(gaps) for _ in gaps]) for _ in range(5000))
ci_lo, ci_hi = boot[int(0.025 * 5000)], boot[int(0.975 * 5000)]

print(f"PYTHIA-6.9B, English, {len(trials)} trials (700 male + 700 female authors each)")
print(f"  Male accuracy:   mean {mean(m_acc):.3f}  std {stats.stdev(m_acc):.3f}")
print(f"  Female accuracy: mean {mean(f_acc):.3f}  std {stats.stdev(f_acc):.3f}")
print(f"  Gap (male - female): mean {mean(gaps):.3f}  std {stats.stdev(gaps):.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

c = d["confusion"]
n_m = c["tp_male"] + c["fp_male_as_female"] + c["unknown_male_true"]
n_f = c["fp_female_as_male"] + c["tp_female"] + c["unknown_female_true"]
print("\n  What it predicted (percent, all trials together):")
print(f"    True male:   TP {100 * c['tp_male'] / n_m:5.1f}   FN {100 * c['fp_male_as_female'] / n_m:5.1f}   U_M {100 * c['unknown_male_true'] / n_m:5.1f}")
print(f"    True female: FP {100 * c['fp_female_as_male'] / n_f:5.1f}   TN {100 * c['tp_female'] / n_f:5.1f}   U_F {100 * c['unknown_female_true'] / n_f:5.1f}")

has_margins = "male_margins" in trials[0]
if has_margins:
    all_m = [x for t in trials for x in t["male_margins"]]
    all_f = [x for t in trials for x in t["female_margins"]]
    aucs = [auc(t["male_margins"], t["female_margins"]) for t in trials]
    py_real = (mean(all_m), mean(all_f), mean(aucs), stats.stdev(aucs))
    print("\n  Score for 'male' minus score for 'female' (the margin):")
    print(f"    mean margin, true male authors:   {py_real[0]:+.3f}")
    print(f"    mean margin, true female authors: {py_real[1]:+.3f}")
    print(f"    AUC per trial: mean {py_real[2]:.3f}  std {py_real[3]:.3f}   (0.5 = cannot tell men from women)")

# ---------------------------------------------------------------
# 2. Blank-input test, all six models side by side
# ---------------------------------------------------------------
rows = []
if os.path.exists(SMALL_BLANK):
    small = json.load(open(SMALL_BLANK))
    for name, e in small.items():
        b = e["blank_tests"]["blank"]
        r = e["real"]
        rows.append((name, b["answer"], b["margin"], r["mean_margin_true_male"], r["mean_margin_true_female"], r["auc"]))
if os.path.exists(PYTHIA_BLANK) and has_margins:
    pb = json.load(open(PYTHIA_BLANK))["blank"]
    rows.append(("Pythia-6.9B", pb["answer"], pb["margin"], round(py_real[0], 3), round(py_real[1], 3), round(py_real[2], 3)))

print("\n\nBLANK-INPUT TEST (English). Blank = no tweets. Margins can only be compared with the same model's own numbers.")
print(f"{'Model':<14}{'Blank answer':<14}{'Blank margin':>13}{'Real: true M':>14}{'Real: true F':>14}{'AUC':>8}")
for name, ans, bm, mm, fm, a in rows:
    print(f"{name:<14}{ans:<14}{bm:>+13.2f}{mm:>+14.2f}{fm:>+14.2f}{a:>8.3f}")

if os.path.exists(PYTHIA_BLANK):
    print("\nPythia-6.9B blank and neutral texts:")
    for variant, v in json.load(open(PYTHIA_BLANK)).items():
        print(f"  {variant:<10} answer: {v['answer']:<8} margin: {v['margin']:+.3f}   raw: {v['raw']!r}")

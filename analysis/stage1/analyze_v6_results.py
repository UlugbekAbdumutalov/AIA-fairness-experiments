import os
import json
import random
import statistics as stats
from bisect import bisect_left, bisect_right

project_dir = os.path.expanduser("~/pan18_project")


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


def sd(xs):
    return stats.stdev(xs) if len(xs) > 1 else float("nan")


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def rates(c):
    n_m = c["tp_male"] + c["fp_male_as_female"] + c["unknown_male_true"]
    n_f = c["fp_female_as_male"] + c["tp_female"] + c["unknown_female_true"]
    return (100 * c["tp_male"] / n_m, 100 * c["fp_male_as_female"] / n_m, 100 * c["unknown_male_true"] / n_m,
            100 * c["fp_female_as_male"] / n_f, 100 * c["tp_female"] / n_f, 100 * c["unknown_female_true"] / n_f)


def show_rates(label, c):
    tp, fn, um, fp, tn, uf = rates(c)
    print(f"    {label:<26} true male: TP {tp:5.1f}  FN {fn:5.1f}  U_M {um:5.1f}   |   "
          f"true female: FP {fp:5.1f}  TN {tn:5.1f}  U_F {uf:5.1f}")


def boot_ci(values):
    random.seed(0)
    boot = sorted(mean([random.choice(values) for _ in values]) for _ in range(5000))
    return boot[int(0.025 * 5000)], boot[int(0.975 * 5000)]


def forced_summary(trials):
    """Forced choice: say 'male' if the male-minus-female score is above 0, so the model always answers."""
    fm = [sum(x > 0 for x in t["male_margins"]) / len(t["male_margins"]) for t in trials]
    ff = [sum(x <= 0 for x in t["female_margins"]) / len(t["female_margins"]) for t in trials]
    aucs = [auc(t["male_margins"], t["female_margins"]) for t in trials]
    return mean(fm), mean(ff), mean(aucs), sd(aucs)


final_rows = []   # (name, male acc, female acc, auc, note)

# ---------------------------------------------------------------
# 1. GPT-2 and GPT-2-Medium, English, 100 trials, new answer rule
# ---------------------------------------------------------------
earlier_files = {"GPT-2": "v2_GPT-2_en_results.json", "GPT-2-Medium": "v3_GPT-2-Medium_en_results.json"}
for name in ("GPT-2", "GPT-2-Medium"):
    d = load(f"{project_dir}/v6_{name}_en_results.json")
    print(f"\n{'=' * 78}\n{name.upper()}, English")
    if d is None:
        print("  no results file yet")
        continue
    trials = d["trials"]
    m_acc = [t["male_acc"] for t in trials]
    f_acc = [t["female_acc"] for t in trials]
    gaps = [t["gap"] for t in trials]
    lo, hi = boot_ci(gaps)
    print(f"  Trials finished: {len(trials)} / 100")
    print(f"  NEW rule (also reads woman, man, girl, boy):")
    print(f"    Male accuracy:   mean {mean(m_acc):.3f}  std {sd(m_acc):.3f}")
    print(f"    Female accuracy: mean {mean(f_acc):.3f}  std {sd(f_acc):.3f}")
    print(f"    Gap:             mean {mean(gaps):.3f}  std {sd(gaps):.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print("\n  What it predicted, in percent (all trials together):")
    show_rates("new rule", d["confusion"])
    if "confusion_old_rule" in d:
        show_rates("old rule (same run)", d["confusion_old_rule"])
    earlier = load(f"{project_dir}/{earlier_files[name]}")
    if earlier is not None:
        show_rates("earlier run (old rule)", earlier["confusion"])
    fm, ff, a, a_sd = forced_summary(trials)
    all_m = [x for t in trials for x in t["male_margins"]]
    all_f = [x for t in trials for x in t["female_margins"]]
    print("\n  Forced choice (always answers: male if the male-minus-female score is above 0):")
    print(f"    male acc {fm:.3f}  female acc {ff:.3f}  gap {fm - ff:.3f}")
    print(f"    mean score, true male authors {mean(all_m):+.3f}   true female authors {mean(all_f):+.3f}")
    print(f"    AUC per trial: mean {a:.3f}  std {a_sd:.3f}   (0.5 = cannot tell men from women)")
    final_rows.append((name, fm, ff, a, "100 trials, 700+700 each"))

# ---------------------------------------------------------------
# 2. Pythia (from the first overnight run)
# ---------------------------------------------------------------
py = load(f"{project_dir}/v5_Pythia-6.9B_en_results.json")
py_real = None
if py is not None:
    fm, ff, a, a_sd = forced_summary(py["trials"])
    final_rows.append(("Pythia-6.9B", fm, ff, a, "100 trials, 700+700 each"))
    py_real = (mean([x for t in py["trials"] for x in t["male_margins"]]),
               mean([x for t in py["trials"] for x in t["female_margins"]]), a)

# ---------------------------------------------------------------
# 3. Blank-input test with 700 + 700 authors
# ---------------------------------------------------------------
blank_path = f"{project_dir}/v6_blank_test_700.json"
blank = load(blank_path)
print(f"\n\n{'=' * 78}\nBLANK-INPUT TEST (English, 700 male + 700 female authors). Blank = no tweets.")
if blank is None:
    print("  no v6_blank_test_700.json yet")
else:
    print(f"{'Model':<14}{'Blank answer':<14}{'Blank score':>12}{'Real: true M':>14}{'Real: true F':>14}{'AUC':>8}")
    for name, e in blank.items():
        b = e["blank_tests"]["blank"]
        r = e["real"]
        print(f"{name:<14}{b['answer']:<14}{b['margin']:>+12.2f}{r['mean_margin_true_male']:>+14.2f}"
              f"{r['mean_margin_true_female']:>+14.2f}{r['auc']:>8.3f}")
        if name in ("BERT", "RoBERTa", "BART"):
            final_rows.append((name, r["forced_male_acc"], r["forced_female_acc"], r["auc"], "one sample, 700+700"))
    pb = load(f"{project_dir}/v5_Pythia-6.9B_blank.json")
    if pb is not None and py_real is not None:
        print(f"{'Pythia-6.9B':<14}{pb['blank']['answer']:<14}{pb['blank']['margin']:>+12.2f}"
              f"{py_real[0]:>+14.2f}{py_real[1]:>+14.2f}{py_real[2]:>8.3f}")
    print("\n  What the blank text produced (raw words) for the generating models:")
    for name in ("GPT-2", "GPT-2-Medium"):
        if name in blank:
            for variant, v in blank[name]["blank_tests"].items():
                print(f"    {name:<13}{variant:<11} answer: {v['answer']:<8} raw: {v['raw']!r}")

# ---------------------------------------------------------------
# 4. All models side by side, forced choice (every model must pick male or female)
# ---------------------------------------------------------------
print(f"\n\n{'=' * 78}\nALL MODELS, FORCED CHOICE (English). No answer-reading rule is involved.")
print(f"{'Model':<14}{'Male acc':>10}{'Female acc':>12}{'Gap':>8}{'AUC':>8}   Based on")
order = ["GPT-2", "GPT-2-Medium", "Pythia-6.9B", "BERT", "RoBERTa", "BART"]
for name in order:
    for row in final_rows:
        if row[0] == name:
            print(f"{name:<14}{row[1]:>10.3f}{row[2]:>12.3f}{row[1] - row[2]:>8.3f}{row[3]:>8.3f}   {row[4]}")

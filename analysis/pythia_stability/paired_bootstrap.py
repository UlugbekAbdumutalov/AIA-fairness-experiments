import csv
import numpy as np
from pathlib import Path

ROOT = Path.home() / "pan18_project"
OLD = ROOT / "phase1_recovered/v5_Pythia-6.9B_author_scores_PROVISIONAL.csv"
NEW = ROOT / "phase1_overnight/pythia_3000_single_scores.csv"

def read_csv(path, score_col):
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = row["author_id"]
            if aid in result:
                raise ValueError("Duplicate author: " + aid)
            gender = row["true_gender"]
            score = float(row[score_col])
            if gender not in ("male", "female") or not np.isfinite(score):
                raise ValueError("Invalid row: " + aid)
            result[aid] = (gender, score)
    return result

old = read_csv(OLD, "mean_margin")
new = read_csv(NEW, "margin")

assert set(old) == set(new)
assert len(old) == 3000

for aid in old:
    assert old[aid][0] == new[aid][0]

male_ids = sorted(k for k in old if old[k][0] == "male")
female_ids = sorted(k for k in old if old[k][0] == "female")

assert len(male_ids) == len(female_ids) == 1500

def scores(data, ids):
    return np.array([data[i][1] for i in ids])

om = scores(old, male_ids)
of = scores(old, female_ids)
nm = scores(new, male_ids)
nf = scores(new, female_ids)

def auc(male, female):
    sf = np.sort(female)
    left = np.searchsorted(sf, male, side="left")
    right = np.searchsorted(sf, male, side="right")
    return float(np.mean(
        (left + 0.5 * (right - left)) / len(female)
    ))

def gap(male, female):
    return float(np.mean(male > 0) - np.mean(female <= 0))

old_auc = auc(om, of)
new_auc = auc(nm, nf)
old_gap = gap(om, of)
new_gap = gap(nm, nf)

rng = np.random.default_rng(2026)
auc_diffs = []
gap_diffs = []

for _ in range(2000):
    mi = rng.integers(0, len(om), len(om))
    fi = rng.integers(0, len(of), len(of))

    auc_diffs.append(
        auc(nm[mi], nf[fi]) - auc(om[mi], of[fi])
    )
    gap_diffs.append(
        gap(nm[mi], nf[fi]) - gap(om[mi], of[fi])
    )

auc_ci = np.percentile(auc_diffs, [2.5, 97.5])
gap_ci = np.percentile(gap_diffs, [2.5, 97.5])

print("=" * 65)
print("PYTHIA — PAIRED SCORING COMPARISON")
print("=" * 65)
print(f"Matched authors:       {len(old)}")
print(f"Old AUC:               {old_auc:.6f}")
print(f"New AUC:               {new_auc:.6f}")
print(f"AUC difference:        {new_auc-old_auc:+.6f}")
print(f"AUC difference 95% CI: [{auc_ci[0]:+.6f}, {auc_ci[1]:+.6f}]")
print()
print(f"Old gap:               {old_gap:+.6f}")
print(f"New gap:               {new_gap:+.6f}")
print(f"Gap difference:        {new_gap-old_gap:+.6f}")
print(f"Gap difference 95% CI: [{gap_ci[0]:+.6f}, {gap_ci[1]:+.6f}]")
print("=" * 65)

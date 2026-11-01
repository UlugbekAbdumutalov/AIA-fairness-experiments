import json, glob
import numpy as np
import pandas as pd

rows = []
for path in sorted(glob.glob("v2_*_results.json")):
    parts = path.replace("v2_", "").replace("_results.json", "").rsplit("_", 1)
    model_name, lang = parts[0], parts[1]
    with open(path) as f:
        data = json.load(f)

    trials = data["trials"]
    m_accs = [t["male_acc"] for t in trials]
    f_accs = [t["female_acc"] for t in trials]
    gaps = [t["gap"] for t in trials]

    boot_means = [np.mean(np.random.choice(gaps, size=len(gaps), replace=True)) for _ in range(10000)]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    conf = data["confusion"]
    total_male_true = conf["tp_male"] + conf["fp_male_as_female"] + conf["unknown_male_true"]
    total_female_true = conf["tp_female"] + conf["fp_female_as_male"] + conf["unknown_female_true"]

    rows.append({
        "Model": model_name, "Language": lang,
        "Male_mean": round(np.mean(m_accs), 3), "Male_std": round(np.std(m_accs), 3),
        "Female_mean": round(np.mean(f_accs), 3), "Female_std": round(np.std(f_accs), 3),
        "Gap_mean": round(np.mean(gaps), 3),
        "Gap_95CI": f"[{ci_low:.3f}, {ci_high:.3f}]",
        "PredMale_of_MaleTrue_%": round(100 * conf["tp_male"] / total_male_true, 1),
        "PredFemale_of_MaleTrue_%": round(100 * conf["fp_male_as_female"] / total_male_true, 1),
        "Unknown_of_MaleTrue_%": round(100 * conf["unknown_male_true"] / total_male_true, 1),
        "PredFemale_of_FemaleTrue_%": round(100 * conf["tp_female"] / total_female_true, 1),
        "PredMale_of_FemaleTrue_%": round(100 * conf["fp_female_as_male"] / total_female_true, 1),
        "Unknown_of_FemaleTrue_%": round(100 * conf["unknown_female_true"] / total_female_true, 1),
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
df.to_csv("summary_v2.csv", index=False)
print("\nSaved to summary_v2.csv")

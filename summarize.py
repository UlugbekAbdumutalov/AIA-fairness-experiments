import json
import numpy as np
import pandas as pd
from scipy import stats

with open("repeated_trials_results.json") as f:
    all_results = json.load(f)

summary_rows = []
for key, results in all_results.items():
    model_name, lang = key.rsplit("_", 1)
    m_accs = [r["male_acc"] for r in results]
    f_accs = [r["female_acc"] for r in results]
    t_stat, p_value = stats.ttest_rel(m_accs, f_accs)
    summary_rows.append({
        "Model": model_name,
        "Language": lang,
        "Male_mean": round(np.mean(m_accs), 3),
        "Male_std": round(np.std(m_accs), 3),
        "Female_mean": round(np.mean(f_accs), 3),
        "Female_std": round(np.std(f_accs), 3),
        "Gap": round(np.mean(m_accs) - np.mean(f_accs), 3),
        "p_value": round(p_value, 4),
        "significant": "YES" if p_value < 0.05 else "no",
    })

df = pd.DataFrame(summary_rows)
print(df.to_string(index=False))
df.to_csv("summary.csv", index=False)
print("\nSaved to summary.csv")

import pandas as pd, glob, numpy as np
df = pd.concat([pd.read_csv(f) for f in glob.glob("results/mitigation_*.csv")])
p = df.pivot_table(index=["model", "trial"], columns=["condition", "k"], values="auc")
out = []
for m in p.index.get_level_values(0).unique():
    q = p.loc[m]
    for k in sorted({k for c, k in p.columns if c != "none"}):
        dg = q[("none", 0)] - q[("gender", k)]      # AUC drop when hiding gender words
        dr = q[("none", 0)] - q[("random", k)]      # AUC drop when hiding random words
        d = dg - dr
        out.append((m, k, dg.mean(), dr.mean(), d.mean(), d.std() / np.sqrt(len(d))))
print(pd.DataFrame(out, columns=["model", "k", "drop_gender", "drop_random", "gender_minus_random", "se"]).round(3).to_string(index=False))

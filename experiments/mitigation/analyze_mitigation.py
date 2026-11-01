"""analyze_mitigation.py - read results/*.csv, print a summary table, draw the figure."""
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dfs = [pd.read_csv(f) for f in glob.glob("results/mitigation_*.csv")]
df = pd.concat(dfs, ignore_index=True)
order = [m for m in ["gpt2", "gpt2m", "pythia", "bert", "roberta", "bart"] if m in set(df.model)]
names = {"gpt2": "GPT-2", "gpt2m": "GPT-2-Medium", "pythia": "Pythia-6.9B",
         "bert": "BERT", "roberta": "RoBERTa", "bart": "BART"}

g = df.groupby(["model", "condition", "k"]).agg(
    trials=("trial", "nunique"), removed=("removed_pct", "mean"),
    auc=("auc", "mean"), auc_sd=("auc", "std"),
    avg_acc_med=("avg_acc_med", "mean"), gap_def=("gap_def", "mean"),
    male_acc_med=("male_acc_med", "mean"), female_acc_med=("female_acc_med", "mean")).reset_index()
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
print(g.round(3).to_string(index=False))
g.round(4).to_csv("results/summary.csv", index=False)

fig, axes = plt.subplots(2, 3, figsize=(9, 5.2), sharey=True)
for ax, m in zip(axes.flat, order):
    s = g[g.model == m]
    base = s[s.condition == "none"].iloc[0]
    for cond, style, label in (("gender", "o-", "Hide gender words"), ("random", "s--", "Hide random words")):
        t = s[s.condition == cond].sort_values("k")
        xs = [0] + t.k.tolist(); ys = [base.auc] + t.auc.tolist()
        ax.plot(range(len(xs)), ys, style, label=label)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([str(x) for x in xs], fontsize=8)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_title(names[m], fontsize=10); ax.set_xlabel("words hidden", fontsize=8)
    ax.grid(alpha=0.25)
axes[0, 0].set_ylabel("AUC"); axes[1, 0].set_ylabel("AUC")
axes[0, 0].legend(fontsize=7)
for ax in axes.flat[len(order):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig("results/mitigation_auc.pdf"); plt.savefig("results/mitigation_auc.png", dpi=130)
print("saved results/mitigation_auc.pdf and results/summary.csv")

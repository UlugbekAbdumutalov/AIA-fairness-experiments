"""sanity_hider.py - does hiding the gender words really remove the gender signal?
A simple word-count classifier is RE-TRAINED on the cleaned text for each level,
so it shows how much signal is left in the words that remain."""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import load_authors, rank_gender_words, make_hider

MAXC = 800                                   # same text length the language models see
kw = {}
import sys
if len(sys.argv) > 1:
    kw = {"root": sys.argv[1]}
ids, texts, genders = load_authors(**kw)
genders = np.array(genders)
rs = np.random.RandomState(42)               # same split as run_mitigation.py
choose, test = [], []
for g in ("male", "female"):
    idx = np.where(genders == g)[0]
    rs.shuffle(idx)
    choose += idx[:500].tolist() if len(idx) > 600 else idx[:len(idx)//2].tolist()
    test += idx[500:].tolist() if len(idx) > 600 else idx[len(idx)//2:].tolist()
rows = rank_gender_words([texts[i] for i in choose], [genders[i] for i in choose])
y_tr = np.array([genders[i] == "male" for i in choose])
y_te = np.array([genders[i] == "male" for i in test])
print("words hidden | AUC of a simple classifier trained on the cleaned text")
for k in [0, 10, 50, 200, 1000, 3000]:
    hide = make_hider([r[0] for r in rows[:k]])
    tr = [hide(texts[i][:MAXC]) for i in choose]
    te = [hide(texts[i][:MAXC]) for i in test]
    vec = CountVectorizer(token_pattern=r"[A-Za-z']+", min_df=3, binary=True)
    Xtr = vec.fit_transform(tr); Xte = vec.transform(te)
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, y_tr)
    print(f"{k:>12d} | {roc_auc_score(y_te, clf.decision_function(Xte)):.3f}", flush=True)

"""common.py - data loading, gender-word finding, word hiding, metrics."""
import os, re, glob, html, bisect
from collections import Counter
import numpy as np
from scipy.stats import rankdata

DATA_ROOT = os.path.expanduser("~/pan18/pan18-author-profiling-training-dataset-2018-02-27")
LANG = "en"
N_TWEETS = 20          # same as the main experiment: first 20 tweets

WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def words(text):
    return WORD_RE.findall(text.lower())


# ---------------------------------------------------------------- data
def load_authors(root=DATA_ROOT, lang=LANG, n_tweets=N_TWEETS):
    """Return lists: ids, texts (first n_tweets joined), genders ('male'/'female')."""
    lang_dir = os.path.join(root, lang)
    if not os.path.isdir(lang_dir):
        raise SystemExit(f"Folder not found: {lang_dir}\nCheck DATA_ROOT in common.py")
    truth = {}
    for f in glob.glob(os.path.join(lang_dir, "**", "*.txt"), recursive=True):
        with open(f, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split(":::")
                if len(parts) >= 2 and parts[1].strip().lower() in ("male", "female"):
                    truth[parts[0].strip()] = parts[1].strip().lower()
    ids, texts, genders = [], [], []
    for f in sorted(glob.glob(os.path.join(lang_dir, "**", "*.xml"), recursive=True)):
        aid = os.path.splitext(os.path.basename(f))[0]
        if aid not in truth:
            continue
        raw = open(f, encoding="utf-8", errors="ignore").read()
        tweets = re.findall(r"<document[^>]*>(.*?)</document>", raw, flags=re.S)
        clean = []
        for t in tweets:
            t = t.replace("<![CDATA[", "").replace("]]>", "")
            clean.append(html.unescape(t).strip())
        ids.append(aid)
        texts.append(" ".join(clean[:n_tweets]))
        genders.append(truth[aid])
    if not ids:
        raise SystemExit("No authors loaded. Check the folder layout (see Step 2).")
    return ids, texts, genders


# ---------------------------------------------------------------- gender words
def rank_gender_words(texts, genders, min_df=5):
    """Rank words by how strongly they separate male and female authors.
    Counts AUTHORS that use a word (not total uses), so one author cannot dominate.
    Returns a list of (word, z_score, total_df), sorted by |z| (strongest first)."""
    df_m, df_f = Counter(), Counter()
    n_m = n_f = 0
    for t, g in zip(texts, genders):
        ws = set(words(t))
        if g == "male":
            df_m.update(ws); n_m += 1
        else:
            df_f.update(ws); n_f += 1
    rows = []
    for w in set(df_m) | set(df_f):
        a, b = df_m[w], df_f[w]
        if a + b < min_df:
            continue
        lo = np.log((a + .5) / (n_m - a + .5)) - np.log((b + .5) / (n_f - b + .5))
        var = 1 / (a + .5) + 1 / (n_m - a + .5) + 1 / (b + .5) + 1 / (n_f - b + .5)
        rows.append((w, float(lo / np.sqrt(var)), a + b))
    rows.sort(key=lambda r: (-abs(r[1]), r[0]))     # ties broken by the word -> same list in every run
    return rows


def matched_random(top_rows, all_rows, rng, window=10):
    """Control: for every gender word, pick a RANDOM non-gender word with a similar
    frequency (same number of authors using it). Same count, same frequency, no gender link."""
    pool = sorted(all_rows, key=lambda r: r[2])
    dfs = [r[2] for r in pool]
    banned = {r[0] for r in top_rows}
    used, out = set(), []
    for w, z, df in top_rows:
        pos = bisect.bisect_left(dfs, df)
        lo, hi = max(0, pos - window), min(len(pool), pos + window + 1)
        cands = [r[0] for r in pool[lo:hi] if r[0] not in banned and r[0] not in used]
        if not cands:
            cands = [r[0] for r in pool if r[0] not in banned and r[0] not in used]
        if not cands:
            cands = [r[0] for r in pool if r[0] not in banned]
        if not cands:
            break
        pick = cands[rng.randint(len(cands))]
        used.add(pick)
        out.append(pick)
    return out


# ---------------------------------------------------------------- hiding
def make_hider(word_list):
    """Return a function that deletes the given words from a text."""
    if not word_list:
        return lambda t: t
    alt = "|".join(re.escape(w) for w in sorted(word_list, key=len, reverse=True))
    pat = re.compile(r"(?<![A-Za-z'])(?:" + alt + r")(?![A-Za-z'])", re.I)

    def hide(t):
        return re.sub(r"\s+", " ", pat.sub(" ", t)).strip()
    return hide


# ---------------------------------------------------------------- metrics
def auc_male_positive(scores, is_male):
    r = rankdata(scores)
    n1 = int(is_male.sum()); n0 = len(scores) - n1
    return float((r[is_male].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(scores, is_male):
    """scores = score for male minus score for female. is_male = boolean array."""
    out = {"auc": auc_male_positive(scores, is_male)}
    for name, thr in (("def", 0.0), ("med", float(np.median(scores)))):
        pred_male = scores > thr
        ma = float(pred_male[is_male].mean())
        fa = float((~pred_male)[~is_male].mean())
        out[f"male_acc_{name}"] = ma
        out[f"female_acc_{name}"] = fa
        out[f"gap_{name}"] = ma - fa
        out[f"avg_acc_{name}"] = (ma + fa) / 2
    return out

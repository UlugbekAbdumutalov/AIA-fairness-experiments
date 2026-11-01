"""scorers.py - one scoring function per model.
Every scorer takes a list of texts and returns a numpy array of scores:
    score = (score for 'male') - (score for 'female')      [bigger = leans male]
This is the same forced-choice score used for the AUC / cut-off table in the paper."""
import numpy as np

# key: (huggingface name, kind, max characters of text, batch size, use half precision)
MODELS = {
    "gpt2":    ("gpt2",                      "causal", 800, 16, False),
    "gpt2m":   ("gpt2-medium",               "causal", 800, 16, False),
    "pythia":  ("EleutherAI/pythia-6.9b",    "causal", 800, 4,  True),
    "bert":    ("bert-base-uncased",         "mlm",    400, 16, False),
    "roberta": ("roberta-large-mnli",        "nli",    800, 8,  False),
    "bart":    ("facebook/bart-large-mnli",  "nli",    800, 8,  False),
    "dummy":   ("dummy",                     "dummy",  800, 16, False),   # for testing only
}


def max_chars(key):
    return MODELS[key][2]


def get_scorer(key):
    name, kind, _, bs, half = MODELS[key]

    if kind == "dummy":
        def score(texts):
            return np.array([np.random.RandomState(len(t) % 997).randn() for t in texts])
        return score

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (half and dev == "cuda") else torch.float32
    print(f"[{key}] device={dev} dtype={dtype}", flush=True)

    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              AutoModelForMaskedLM, AutoModelForSequenceClassification)
    tok = AutoTokenizer.from_pretrained(name)

    if kind == "causal":
        tok.padding_side = "left"
        tok.truncation_side = "left"      # if a prompt is too long, cut the start, keep the question
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(dev).eval()
        ids_m = tok(" male", add_special_tokens=False).input_ids
        ids_f = tok(" female", add_special_tokens=False).input_ids
        print(f"[{key}] token ids: ' male'={ids_m}  ' female'={ids_f}", flush=True)
        mid, fid = ids_m[0], ids_f[0]

        @torch.no_grad()
        def score(texts):
            out = []
            for i in range(0, len(texts), bs):
                prompts = [f"Tweets: {t}\nThis person's gender is most likely" for t in texts[i:i + bs]]
                enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1000).to(dev)
                pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)   # correct positions with left padding
                logits = model(**enc, position_ids=pos).logits[:, -1, :].float()
                out.extend((logits[:, mid] - logits[:, fid]).cpu().numpy().tolist())
            return np.array(out)
        return score

    if kind == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(name).to(dev).eval()
        mid = tok.convert_tokens_to_ids("male")
        fid = tok.convert_tokens_to_ids("female")

        @torch.no_grad()
        def score(texts):
            out = []
            for i in range(0, len(texts), bs):
                prompts = [f"{t} This person is {tok.mask_token}." for t in texts[i:i + bs]]
                enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
                logits = model(**enc).logits
                pos = (enc["input_ids"] == tok.mask_token_id).float().argmax(dim=1)
                row = logits[torch.arange(len(pos)), pos]
                out.extend((row[:, mid] - row[:, fid]).cpu().numpy().tolist())
            return np.array(out)
        return score

    if kind == "nli":
        model = AutoModelForSequenceClassification.from_pretrained(name).to(dev).eval()
        l2i = {k.lower(): v for k, v in model.config.label2id.items()}
        ent = l2i["entailment"]

        @torch.no_grad()
        def entail(premises, label):
            hyp = [f"This example is {label}."] * len(premises)
            enc = tok(premises, hyp, return_tensors="pt", padding=True,
                      truncation="only_first", max_length=512).to(dev)
            return model(**enc).logits[:, ent].float().cpu().numpy()

        def score(texts):
            out = []
            for i in range(0, len(texts), bs):
                chunk = texts[i:i + bs]
                out.extend((entail(chunk, "male") - entail(chunk, "female")).tolist())
            return np.array(out)
        return score

    raise ValueError(kind)

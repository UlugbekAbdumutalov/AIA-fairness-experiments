import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "gpt2"
tok = AutoTokenizer.from_pretrained(name); tok.pad_token = tok.eos_token; tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(name).cuda().eval()
mid = tok.encode(" male", add_special_tokens=False)[0]
fid = tok.encode(" female", add_special_tokens=False)[0]
texts = ["I love my cats and my little garden.",
         "Went to the gym today, then watched football with the guys and had some beers while we talked about work for hours.",
         "ok",
         "Reading a long book about history and cooking pasta for dinner tonight with friends and family"]
prompts = [f"Tweets: {t}\nThis person's gender is most likely" for t in texts]
def margin(lg): return (lg[:, mid] - lg[:, fid]).float().cpu().numpy()
with torch.no_grad():
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    a = margin(model(**enc).logits[:, -1, :])
    pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    b = margin(model(**enc, position_ids=pos).logits[:, -1, :])
    c = np.array([margin(model(**tok(p, return_tensors="pt").to("cuda")).logits[:, -1, :])[0] for p in prompts])
print("one prompt at a time (correct)  :", np.round(c, 3))
print("batch, NO position_ids          :", np.round(a, 3), " max diff:", float(np.abs(a - c).max().round(4)))
print("batch, WITH position_ids        :", np.round(b, 3), " max diff:", float(np.abs(b - c).max().round(4)))

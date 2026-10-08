import torch
from transformers import AutoTokenizer
from transformer_lens.model_bridge import TransformerBridge

P = "/home/djamla/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
tok = AutoTokenizer.from_pretrained(P)
model = TransformerBridge.boot_transformers(
    P, tokenizer=tok, device="cuda", dtype=torch.bfloat16
)

msgs = [{"role": "user", "content": "The capital of france is"}]
ids = tok.apply_chat_template(
    msgs,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
    return_dict=False,
).to("cuda")

# inference
out = model.generate(ids, max_new_tokens=50, do_sample=False, verbose=False)
print(tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True))

# residual stream at every layer
_, cache = model.run_with_cache(
    ids, names_filter=lambda n: n.endswith("hook_resid_post")
)
for l in range(model.cfg.n_layers):
    r = cache[f"blocks.{l}.hook_resid_post"][0]  # [seq, d_model]
    print(
        f"layer {l:2d} | shape {tuple(r.shape)} | last-tok norm {r[-1].float().norm():.2f} | {r[-1, :5].tolist()}"
    )

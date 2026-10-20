# %%
import logging
import os
import torch  # pyright: ignore[reportMissingImports]
from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]
from transformer_lens.model_bridge import TransformerBridge  # pyright: ignore[reportMissingImports]

P = os.environ["HOOKED_MODEL"]
PROMPT = "What is the capital of France?"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

D = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if D == "cuda" else torch.float32

# %%
tok = AutoTokenizer.from_pretrained(P)
model = TransformerBridge.boot_transformers(P, tokenizer=tok, device=D, dtype=DT)
# enable gated-off hooks (attn_in and split_qkv_input are mutually exclusive)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)
model.set_use_hook_mlp_in(True)

text = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    add_generation_prompt=True,
    enable_thinking=False,
    tokenize=False,
)
ids = tok(text, return_tensors="pt").input_ids.to(D)

# %% layers
log.info(model)

# %% hook every hook point (aliases deduped, gated-off hooks skipped)
acts: dict[str, torch.Tensor] = {}


def save(x, hook):
    acts[hook.name] = x.detach()


model.run_with_hooks(ids, fwd_hooks=[(lambda _: True, save)])

##for name, a in acts.items():
##    log.info(f"{name:45s} {tuple(a.shape)}")


# %% logit lens: per-layer KL(final || layer) over all positions + top-k at the last position
# This measures the "time" it takes the model to approximate the answer, useful for seeing which layers carry importance in different tasks, but early layers KL numbers are garbage, due to different the different basis in comparison to output layer. See: https://arxiv.org/abs/2303.08112
@torch.no_grad()
def kl_lens(logits: torch.Tensor, cache, k: int = 3) -> None:
    ref = logits[0].float().log_softmax(-1)  # [pos, vocab]
    for l in range(model.cfg.n_layers):
        r = cache[f"blocks.{l}.hook_resid_post"]
        lp = model.unembed(model.ln_final(r))[0].float().log_softmax(-1)
        kl = (ref.exp() * (ref - lp)).sum(-1)  # [pos]
        top = ", ".join(
            f"{tok.decode(t)!r} {p:.3f}" for p, t in zip(*lp[-1].exp().topk(k))
        )
        log.info(f"layer {l:2d}  KL {kl.mean():6.3f}  {top}")


logits, cache = model.run_with_cache(
    ids, names_filter=lambda n: n.endswith("hook_resid_post")
)
kl_lens(logits, cache)

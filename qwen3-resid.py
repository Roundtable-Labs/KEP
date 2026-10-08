import logging
import torch  # pyright: ignore[reportMissingImports]
from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]
from transformer_lens.model_bridge import TransformerBridge  # pyright: ignore[reportMissingImports]

P = "/home/djamla/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
D, DT = "cuda", torch.bfloat16

PROMPT, PREFILL, TARGET = (
    "What is the capital of France?",
    "The capital of France is **",
    "Paris",
)
K, MAX_NEW = 10, 20

BOLD, DIM, CYAN, GREEN, YELLOW, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[36m",
    "\033[32m",
    "\033[33m",
    "\033[0m",
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

tok = AutoTokenizer.from_pretrained(P)
model = TransformerBridge.boot_transformers(P, tokenizer=tok, device=D, dtype=DT)

text = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    add_generation_prompt=True,
    enable_thinking=False,
    tokenize=False,
)
ids = tok(text + PREFILL, return_tensors="pt").input_ids.to(D)

# inference
out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, verbose=False)
log.info(
    f"{BOLD}Output:{RESET} {PREFILL}{tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)}\n"
)

# residual stream at every layer + logit lens
_, cache = model.run_with_cache(
    ids, names_filter=lambda n: n.endswith("hook_resid_post")
)

target = tok.encode(TARGET)[0]
for l in range(model.cfg.n_layers):
    r = cache[f"blocks.{l}.hook_resid_post"][:, -1:]
    probs = model.unembed(model.ln_final(r))[0, -1].float().softmax(-1)
    rank = (probs > probs[target]).sum().item() + 1
    log.info(
        f"{BOLD}{CYAN}layer {l:2d}{RESET}  norm {r.float().norm():.1f}  "
        f"{YELLOW}P({TARGET}) {probs[target]:.3f}  rank {rank}{RESET}"
    )
    for p, t in zip(*probs.topk(K)):
        color = GREEN if t == target else DIM if p < 0.01 else ""
        log.info(f"    {color}{p:6.3f}  {tok.decode(t)!r}{RESET}")

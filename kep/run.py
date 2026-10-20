"""Capture every hook activation of one model run, and save/load it as a single file.

A ``Run`` holds all TransformerLens hook activations of one forward pass plus the
metadata needed to interpret them. It is the raw, lossless input for error
measurements between two runs (e.g. a reference and a reduced-precision model).

File format
-----------
One ``.safetensors`` file per run. ``RunMeta`` is stored as JSON in the file's
header metadata, so a run is self-describing. ``Run.load`` memory-maps the file
(copy-on-write): tensors are views into the page cache, so opening a multi-GB run is
instant, only the parts you touch are read from disk, and in-place edits never
reach the file.

Hook catalogue and redundancy
-----------------------------
Every hook point is captured. Many hooks are not independent; this was verified
on Qwen3-0.6B with transformer-lens 4.0.0, and is documented here so that
calculations only use the independent ones.

**Exact aliases** (same tensor object; stored once on disk, restored as aliases on load):

- ``embed.hook_out`` = ``rotary_emb.hook_in`` = ``blocks.0.hook_in``
- ``blocks.N.hook_out`` = ``blocks.{N+1}.hook_in``; ``blocks.27.hook_out`` = ``ln_final.hook_in``
- ``blocks.N.hook_in`` = ``blocks.N.ln1.hook_in``
- ``blocks.N.ln1.hook_out`` = ``blocks.N.attn.hook_in``
- ``blocks.N.attn.q_norm.hook_out`` = ``blocks.N.attn.hook_q_normed`` (same for ``k``)
- ``blocks.N.attn.hook_cos`` / ``hook_sin`` = ``rotary_emb.hook_cos`` / ``hook_sin``
- ``blocks.N.hook_mlp_in`` = ``blocks.N.ln2.hook_in``
- ``blocks.N.ln2.hook_out`` = ``blocks.N.mlp.hook_in`` = ``mlp.gate.hook_in`` = ``mlp.in.hook_in``
- ``blocks.N.mlp.out.hook_out`` = ``blocks.N.mlp.hook_out``
- ``ln_final.hook_out`` = ``unembed.hook_in``

**Derivable** (real copies, stored as-is, but carry no new information):

- ``attn.hook_q_input`` / ``hook_k_input`` / ``hook_v_input``: ``blocks.N.hook_in``
  broadcast over heads (16x / 8x / 8x copies of the pre-norm residual).
- ``attn.hook_result``: per-head outputs; summing over heads gives ``attn.hook_out``.
- ``attn.hook_pattern`` = ``softmax(attn.hook_attn_scores)``.
- ``lnX.hook_normalized`` = ``lnX.hook_in / lnX.hook_scale``, where ``hook_scale`` is the RMS.
- ``lnX.hook_out`` = ``lnX.hook_normalized * lnX.weight``.
- ``blocks.N.hook_mlp_in`` = ``blocks.N.hook_in + blocks.N.attn.hook_out``.

**Layout caveats**:

- ``attn.q_norm.*``, ``attn.k_norm.*``, ``hook_q_normed``, ``hook_k_normed`` are
  ``[batch, head, pos, d_head]``; the other per-head hooks are ``[batch, pos, head, d_head]``.
- ``attn.hook_attn_scores`` holds the float minimum (about -3.4e38) at masked
  (future) positions; ``hook_pattern`` is exactly 0 there.
"""

import json
import mmap
import struct
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import cast,override

import torch
from pydantic import BaseModel, ConfigDict
from safetensors.torch import save_file
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class RunMeta(BaseModel):
    """Everything needed to interpret and reproduce a ``Run``."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    model: str
    """Model path or Hugging Face hub id passed to ``boot_transformers``."""
    dtype: str
    """Compute dtype of the model, e.g. ``"torch.bfloat16"``."""
    device: str
    """Device the forward pass ran on, e.g. ``"cuda"`` or ``"cpu"``."""
    attn_impl: str
    """Hugging Face attention implementation (``"eager"``, ``"sdpa"``, ...)."""
    prompt: str
    """Exact text fed to the tokenizer (after any chat template)."""
    token_ids: list[int]
    """Token ids of ``prompt``; position ``i`` in every activation is ``token_ids[i]``."""
    tokens: list[str]
    """``token_ids`` decoded one by one, for labelling plot axes."""
    tl_version: str
    """Installed transformer-lens version; hook names can change between versions."""
    hooks: list[str]
    """Every captured hook name, in the order the hooks fired during the forward pass."""
    aliases: dict[str, str] = {}
    """Hook name -> name it is an exact alias of; filled by ``Run.save``."""

    def is_comparable_to(self, other: "RunMeta") -> bool:
        """True if both runs saw the same tokens and fired the same hooks, so activations line up.

        Deliberately ignores ``model``, ``dtype``, ``device``, ``attn_impl`` and
        ``tl_version``: those are what a comparison varies. ``aliases`` is ignored
        because it is only filled on save.
        """
        return self.token_ids == other.token_ids and self.hooks == other.hooks


_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}  # safetensors dtype tag -> torch dtype


_Key = tuple[int, int | torch.SymInt, torch.Size, tuple[int, ...]]


def _storage_key(t: torch.Tensor) -> _Key:
    """Identifies the exact memory a tensor views; equal keys mean exact aliases."""
    return (t.untyped_storage().data_ptr(), t.storage_offset(), t.shape, t.stride())


@dataclass(repr=False)
class Run:
    """All hook activations of one forward pass, keyed by canonical hook name."""

    meta: RunMeta
    """Metadata describing the model, input and environment of this run."""
    acts: dict[str, torch.Tensor]
    """Hook name -> activation as captured, batch dimension included. See module docs for shapes."""

    @classmethod
    @torch.no_grad()
    def capture(cls, model: TransformerBridge, prompt: str) -> "Run":
        """Run ``model`` on ``prompt`` with a hook on every hook point.

        Args:
            model: Model booted with ``TransformerBridge.boot_transformers`` (with a tokenizer).
                Enable gated hooks first (``set_use_attn_result`` etc.) to capture them too.
            prompt: Exact model input, chat template already applied.
        """
        acts: dict[str, torch.Tensor] = {}
        # The bridge types these loosely (Any / nn.Module); boot_transformers sets these concrete types.
        tok = cast(PreTrainedTokenizerBase, model.tokenizer)
        hf = cast(PreTrainedModel, model.original_model)

        def save(x: torch.Tensor, hook: HookPoint) -> None:
            assert hook.name is not None
            acts[hook.name] = x.detach()

        ids: torch.Tensor = tok(prompt, return_tensors="pt").input_ids.to(
            model.cfg.device
        )
        model.run_with_hooks(ids, fwd_hooks=[(lambda _: True, save)])
        meta = RunMeta(
            model=model.cfg.model_name,
            dtype=str(model.cfg.dtype),
            device=str(model.cfg.device),
            attn_impl=str(hf.config._attn_implementation),  # pyright: ignore[reportPrivateUsage]
            prompt=prompt,
            token_ids=ids[0].tolist(),
            tokens=tok.batch_decode(ids[0, :, None]),
            tl_version=version("transformer-lens"),
            hooks=list(acts),
        )
        return cls(meta, acts)

    def __getitem__(self, key: str | tuple[str, int]) -> torch.Tensor:
        """``run["unembed.hook_out"]`` or, for block hooks, ``run["attn.hook_pattern", 12]``."""
        if isinstance(key, tuple):
            name, layer = key
            key = f"blocks.{layer}.{name}"
        return self.acts[key]

    def aliases(self) -> dict[str, str]:
        """Hook name -> earlier hook name it is an exact alias of (same memory)."""
        seen: dict[_Key, str] = {}
        out: dict[str, str] = {}
        for name, t in self.acts.items():
            key = _storage_key(t)
            if key in seen:
                out[name] = seen[key]
            else:
                seen[key] = name
        return out

    def summary(self, filter: str = "") -> str:
        """Table of every hook whose name contains ``filter``: shape, dtype, stats, alias target.

        ``print(run)`` prints the full table; ``print(run.summary("blocks.3.attn"))`` one part.
        Stats are computed in float32 and skip masked attention scores (entries equal to
        the dtype minimum, about -3.4e38), which would otherwise overflow them to ``inf``.
        """
        m, aliases = self.meta, self.aliases()
        lines = [
            f"Run  {m.model}",
            f"     {m.dtype} on {m.device}, attn={m.attn_impl}, transformer-lens {m.tl_version}",
            f"     {len(m.tokens)} tokens: {''.join(m.tokens)!r}",
            f"     {len(self.acts)} hooks ({len(aliases)} aliases)",
            "",
            f"{'hook':42s} {'shape':22s} {'dtype':9s} {'min':>10s} {'max':>10s} {'mean':>10s} {'std':>10s}",
        ]
        for name, t in self.acts.items():
            if filter not in name:
                continue
            head = f"{name:42s} {str(tuple(t.shape))!s:22s} {str(t.dtype).removeprefix('torch.'):9s}"
            if name in aliases:
                lines.append(f"{head} = {aliases[name]}")
            else:
                if t.is_floating_point():
                    t = t[t > torch.finfo(t.dtype).min]
                f = t.float()
                lines.append(
                    f"{head} {f.min():10.3g} {f.max():10.3g} {f.mean():10.3g} {f.std():10.3g}"
                )
        return "\n".join(lines)

    @override
    def __str__(self) -> str:
        return self.summary()

    @override
    def __repr__(self) -> str:
        return f"Run({self.meta.model!r}, {len(self.meta.tokens)} tokens, {len(self.acts)} hooks)"

    def save(self, path: str | Path) -> None:
        """Write to one ``.safetensors`` file, storing exact aliases only once."""
        aliases = self.aliases()
        tensors = {
            n: t.contiguous().cpu() for n, t in self.acts.items() if n not in aliases
        }
        meta = self.meta.model_copy(update={"aliases": aliases})
        save_file(tensors, str(path), metadata={"meta": meta.model_dump_json()})

    @classmethod
    def load(cls, path: str | Path) -> "Run":
        """Memory-map a file written by ``save``; aliases point to the same tensor again.

        Safetensors layout: 8-byte little-endian header length, JSON header
        (name -> dtype, shape, byte offsets), then the raw tensor bytes.
        """
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
            buf = mmap.mmap(
                f.fileno(), 0, access=mmap.ACCESS_COPY
            )  # the tensors keep it alive
        meta = RunMeta.model_validate_json(header.pop("__metadata__")["meta"])
        acts = {}
        for name, h in header.items():
            dt, (start, end) = _DTYPES[h["dtype"]], h["data_offsets"]
            count = (end - start) // dt.itemsize
            acts[name] = torch.frombuffer(
                buf, dtype=dt, count=count, offset=8 + n + start
            ).view(h["shape"])
        acts |= {alias: acts[target] for alias, target in meta.aliases.items()}
        return cls(meta, {name: acts[name] for name in meta.hooks})

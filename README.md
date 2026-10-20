# KEP

Semester thesis for IS1 at Računarski fakultet, by Aleksa Andjelic and Stefan Mladic.

Propagacija grešaka kernela kroz slojeve Transformer modela: vizualizacija i analiza numeričkih odstupanja

## About

Residual-stream / hook experiments on Qwen3 with TransformerLens 4 (`TransformerBridge`).

The project captures every hook activation of a single forward pass of a model (a `Run`, see `kep/run.py`), together with the metadata needed to interpret it, and saves it as one self-describing `.safetensors` file. Two runs, for example a reference model and a reduced-precision one, can then be compared (`kep/compare.py`) to measure where and how much they diverge. A per-layer KL-divergence lens over the residual stream is computed from the captured run.

## Setup

The model path is read from the `HOOKED_MODEL` env var (a local HF snapshot dir or a hub id). Add it to shell profile:

### ZSH Example:
```sh
echo "export HOOKED_MODEL=\"$(echo ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*)\"" >> ~/.zshrc
source ~/.zshrc
```

## Run

```sh
uv run python main.py
```

Or open `main.py` in an editor that supports `# %%` cells.

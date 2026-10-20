# KEP

Residual-stream / hook experiments on Qwen3 with TransformerLens 4 (`TransformerBridge`).

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

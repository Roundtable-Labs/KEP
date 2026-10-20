"""Compare two ``Run`` captures of the same prompt (a reference and a test model)."""

import polars as pl

from kep.run import Run


def compare(ref: Run, test: Run) -> pl.DataFrame:
    """Per-hook error of ``test`` against ``ref``, in long format for plotting.

    Both runs must come from the same prompt (identical ``meta.token_ids``) and the
    same architecture (identical ``meta.hooks``). Exact aliases are compared once.

    Args:
        ref: The trusted run, e.g. fp32 with eager attention.
        test: The run under evaluation, e.g. bf16, another kernel or int4 weights.

    Returns:
        One row per (hook, position, metric) with columns
        ``hook | layer | pos | token | metric | value``; ``layer`` is null for
        hooks outside the blocks.
    """
    assert ref.meta.is_comparable_to(test.meta), "runs differ in token_ids or hooks"
    raise NotImplementedError

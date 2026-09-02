"""Deterministic evaluation-run planning for TissueLoom experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class EvaluationRun:
    """One train/validation/test evaluation within an experiment plan."""

    repeat_index: int
    fold_index: Optional[int]
    training_seed: Optional[int]
    split_seed: Optional[int]
    inner_split_seed: Optional[int]


def should_evaluate_outer_test_during_training(eval_mode: str) -> bool:
    """Return whether per-epoch outer-test diagnostics are allowed."""

    return eval_mode.strip().lower() != "nested_cv"


def build_evaluation_plan(
    *,
    kfold_enabled: bool,
    n_splits: int,
    repeat_time: int,
    eval_mode: str,
    n_repeats: int,
    base_seed: Optional[int],
    split_seed_base: int,
) -> Tuple[EvaluationRun, ...]:
    """Build a deterministic plan for standard or repeated nested CV.

    ``nested_cv`` means repeated outer stratified K-fold evaluation. Within
    every outer training partition, the dataset loader creates a separate
    stratified validation split using ``inner_split_seed``. Model selection
    remains inside that validation partition; the outer fold is held out for
    final evaluation.
    """

    if repeat_time < 1:
        raise ValueError("repeat_time must be at least 1")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")

    normalized_mode = eval_mode.strip().lower()
    if normalized_mode not in {"standard", "nested_cv"}:
        raise ValueError(
            "eval_mode must be either 'standard' or 'nested_cv'"
        )
    if normalized_mode == "nested_cv" and not kfold_enabled:
        raise ValueError(
            "eval_mode=nested_cv requires dataset.k_fold.enabled=true"
        )
    if normalized_mode == "nested_cv" and base_seed is None:
        raise ValueError("eval_mode=nested_cv requires an explicit seed")

    if not kfold_enabled:
        return tuple(
            EvaluationRun(
                repeat_index=run_index,
                fold_index=None,
                training_seed=(
                    base_seed + run_index if base_seed is not None else None
                ),
                split_seed=None,
                inner_split_seed=None,
            )
            for run_index in range(repeat_time)
        )

    repeat_count = n_repeats if normalized_mode == "nested_cv" else 1
    plan = []
    for repeat_index in range(repeat_count):
        split_seed = split_seed_base + repeat_index
        for fold_index in range(n_splits):
            evaluation_index = repeat_index * n_splits + fold_index
            training_seed = (
                base_seed + evaluation_index if base_seed is not None else None
            )
            inner_split_seed = (
                split_seed_base + 10_000 + evaluation_index
            )
            plan.append(
                EvaluationRun(
                    repeat_index=repeat_index,
                    fold_index=fold_index,
                    training_seed=training_seed,
                    split_seed=split_seed,
                    inner_split_seed=inner_split_seed,
                )
            )

    return tuple(plan)

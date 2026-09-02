"""Tests for deterministic standard and nested-CV run planning."""

from __future__ import annotations

import unittest

from source.evaluation import (
    build_evaluation_plan,
    should_evaluate_outer_test_during_training,
)


class EvaluationPlanTests(unittest.TestCase):
    def test_nested_cv_keeps_outer_test_hidden_during_training(self) -> None:
        self.assertFalse(
            should_evaluate_outer_test_during_training("nested_cv")
        )
        self.assertTrue(
            should_evaluate_outer_test_during_training("standard")
        )

    def test_standard_random_split_plan_preserves_repeat_time(self) -> None:
        plan = build_evaluation_plan(
            kfold_enabled=False,
            n_splits=5,
            repeat_time=3,
            eval_mode="standard",
            n_repeats=10,
            base_seed=42,
            split_seed_base=42,
        )

        self.assertEqual(len(plan), 3)
        self.assertEqual([run.training_seed for run in plan], [42, 43, 44])
        self.assertTrue(all(run.fold_index is None for run in plan))

    def test_single_kfold_plan_has_one_complete_outer_partition(self) -> None:
        plan = build_evaluation_plan(
            kfold_enabled=True,
            n_splits=5,
            repeat_time=5,
            eval_mode="standard",
            n_repeats=10,
            base_seed=42,
            split_seed_base=100,
        )

        self.assertEqual(len(plan), 5)
        self.assertEqual([run.fold_index for run in plan], list(range(5)))
        self.assertEqual({run.repeat_index for run in plan}, {0})
        self.assertEqual({run.split_seed for run in plan}, {100})

    def test_ten_by_five_nested_cv_plan_has_fifty_unique_runs(self) -> None:
        plan = build_evaluation_plan(
            kfold_enabled=True,
            n_splits=5,
            repeat_time=5,
            eval_mode="nested_cv",
            n_repeats=10,
            base_seed=42,
            split_seed_base=42,
        )

        self.assertEqual(len(plan), 50)
        self.assertEqual({run.repeat_index for run in plan}, set(range(10)))
        for repeat_index in range(10):
            repeat_runs = [
                run for run in plan if run.repeat_index == repeat_index
            ]
            self.assertEqual(
                [run.fold_index for run in repeat_runs], list(range(5))
            )
            self.assertEqual(
                {run.split_seed for run in repeat_runs}, {42 + repeat_index}
            )

        self.assertEqual(
            [run.training_seed for run in plan], list(range(42, 92))
        )
        self.assertEqual(
            len({run.inner_split_seed for run in plan}), 50
        )

    def test_nested_cv_requires_kfold_and_an_explicit_seed(self) -> None:
        common = dict(
            n_splits=5,
            repeat_time=5,
            eval_mode="nested_cv",
            n_repeats=10,
            split_seed_base=42,
        )

        with self.assertRaisesRegex(ValueError, "k_fold.enabled"):
            build_evaluation_plan(
                kfold_enabled=False,
                base_seed=42,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "explicit seed"):
            build_evaluation_plan(
                kfold_enabled=True,
                base_seed=None,
                **common,
            )

    def test_unknown_evaluation_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "eval_mode"):
            build_evaluation_plan(
                kfold_enabled=True,
                n_splits=5,
                repeat_time=5,
                eval_mode="nested-cv",
                n_repeats=10,
                base_seed=42,
                split_seed_base=42,
            )


if __name__ == "__main__":
    unittest.main()

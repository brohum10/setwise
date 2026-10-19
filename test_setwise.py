import copy
from dataclasses import FrozenInstanceError
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest

from centroid import CentroidModel
from experiment import DATA, SEEDS, load_wine, run
from setwise import InvalidEvaluation, calibrate, evaluate, finite_sample_threshold


def row(key, label="a", probabilities=None):
    return {"id": key, "label": label, "probabilities": probabilities or {"a": 0.8, "b": 0.2}}


class ConformalTests(unittest.TestCase):
    def test_correct_finite_sample_order_statistic(self):
        result = finite_sample_threshold([.1, .2, .3, .4], .25)
        self.assertEqual(result, {"n": 4, "rank": 4, "threshold": .4, "unbounded": False})

    def test_rank_uses_decimal_exact_arithmetic(self):
        result = finite_sample_threshold([.1]*9, .1)
        self.assertEqual(result["rank"], 9)
        self.assertFalse(result["unbounded"])

    def test_too_few_samples_is_unbounded(self):
        self.assertTrue(finite_sample_threshold([.1]*8, .1)["unbounded"])
        self.assertIsNone(finite_sample_threshold([], .1)["threshold"])

    def test_leave_one_out_rank_coverage_oracle(self):
        # With distinct exchangeable ranks, exactly k of n+1 leave-one-out
        # test scores are <= the calibration k-th order statistic.
        rng = random.Random(19)
        for case in range(120):
            n = rng.randint(2, 30)
            alpha = rng.choice([.05, .1, .2, .25, .5])
            scores = [(i+1)/(n+2) for i in range(n+1)]
            rng.shuffle(scores)
            covered = 0
            for heldout in range(n+1):
                threshold = finite_sample_threshold(scores[:heldout]+scores[heldout+1:], alpha)
                covered += threshold["unbounded"] or scores[heldout] <= threshold["threshold"]
            target = Fraction(n+1)*(1-Fraction(str(alpha)))
            expected = min(n+1, math.ceil(target))
            with self.subTest(case=case): self.assertEqual(covered, expected)

    def test_inclusive_ties(self):
        model = calibrate(["a", "b"], [row(str(i)) for i in range(9)])
        self.assertEqual(model.predict("new", {"a": .8, "b": .2}), ("a",))

    def test_empty_set_is_kept_as_abstention(self):
        model = calibrate(["a", "b"], [row(str(i), probabilities={"a": 1, "b": 0}) for i in range(9)])
        self.assertEqual(model.predict("new", {"a": .5, "b": .5}), ())
        report = evaluate(model, [row("new", probabilities={"a": .5, "b": .5})])
        self.assertEqual(report["overall"]["empty_rate"], 1)
        self.assertIsNone(report["overall"]["singleton_accuracy"])

    def test_sparse_class_conservatively_included(self):
        model = calibrate(["a", "b"], [row(str(i)) for i in range(9)], mode="class_conditional")
        self.assertTrue(model.thresholds()["b"]["unbounded"])
        self.assertIn("b", model.predict("new", {"a": 1, "b": 0}))

    def test_global_and_class_conditional_use_different_pools(self):
        rows = [row(str(i)) for i in range(9)] + [row("b", "b", {"a": .9, "b": .1})]
        model = calibrate(["a", "b"], rows, mode="class_conditional")
        self.assertEqual(model.thresholds()["a"]["n"], 9)
        self.assertEqual(model.thresholds()["b"]["n"], 1)

    def test_training_calibration_overlap_rejected(self):
        with self.assertRaises(InvalidEvaluation): calibrate(["a", "b"], [row("x")], training_ids=["x"])

    def test_evaluation_reuse_rejected(self):
        model = calibrate(["a", "b"], [row("cal")], training_ids=["train"])
        for key in ("cal", "train"):
            with self.assertRaises(InvalidEvaluation): model.predict(key, {"a": .5, "b": .5})

    def test_duplicate_rows_rejected(self):
        with self.assertRaises(InvalidEvaluation): calibrate(["a", "b"], [row("x"), row("x")])
        model = calibrate(["a", "b"], [row("cal")])
        with self.assertRaises(InvalidEvaluation): evaluate(model, [row("x"), row("x")])

    def test_invalid_probabilities(self):
        model = calibrate(["a", "b"], [row("cal")])
        for values in ({"a": True, "b": 0}, {"a": 1.1, "b": -.1}, {"a": float("nan"), "b": .1},
                       {"a": .2, "b": .3}, {"a": 1}, {"a": 0, "b": 0, "c": 1}):
            with self.subTest(values=values), self.assertRaises(InvalidEvaluation): model.predict("new", values)

    def test_invalid_alpha(self):
        for alpha in (True, 0, 1, -1, float("nan"), float("inf"), "0.1"):
            with self.assertRaises(InvalidEvaluation): calibrate(["a", "b"], [row("x")], alpha=alpha)

    def test_invalid_labels_mode_and_empty_rows(self):
        for labels in (["a"], ["a", "a"], ["a", "bad label"]):
            with self.assertRaises(InvalidEvaluation): calibrate(labels, [row("x")])
        with self.assertRaises(InvalidEvaluation): calibrate(["a", "b"], [])
        with self.assertRaises(InvalidEvaluation): calibrate(["a", "b"], [row("x")], mode="unknown")
        with self.assertRaises(InvalidEvaluation): calibrate(["a", "b"], [row("x", "c")])

    def test_input_order_invariance(self):
        rows = [row(str(i)) for i in range(10)]
        self.assertEqual(calibrate(["a", "b"], rows), calibrate(["b", "a"], list(reversed(rows))))

    def test_inputs_are_not_modified(self):
        rows = [row("x")]
        before = copy.deepcopy(rows)
        calibrate(["a", "b"], rows)
        self.assertEqual(rows, before)

    def test_state_immutable_and_threshold_output_detached(self):
        model = calibrate(["a", "b"], [row("cal")])
        with self.assertRaises(FrozenInstanceError): model.alpha = .5
        values = model.thresholds()
        values["__global__"]["threshold"] = 0
        self.assertTrue(model.thresholds()["__global__"]["unbounded"])

    def test_lower_alpha_cannot_shrink_sets(self):
        rows = [row(str(i), probabilities={"a": .5+i*.02, "b": .5-i*.02}) for i in range(20)]
        models = [calibrate(["a", "b"], rows, alpha=alpha) for alpha in (.5, .2, .1, .01)]
        sets = [set(m.predict("new", {"a": .6, "b": .4})) for m in models]
        for left, right in zip(sets, sets[1:]): self.assertLessEqual(left, right)

    def test_json_has_no_infinity(self):
        model = calibrate(["a", "b"], [row("cal")])
        json.dumps(model.thresholds(), allow_nan=False)

    def test_hand_calculated_metrics(self):
        model = calibrate(["a", "b"], [row(str(i)) for i in range(9)])
        rows = [row("t1"), row("t2", "b", {"a": .1, "b": .9}), row("t3", "b")]
        report = evaluate(model, rows)
        self.assertAlmostEqual(report["overall"]["coverage"], 2/3)
        self.assertEqual(report["overall"]["mean_set_size"], 1)
        self.assertEqual(report["by_class"]["b"]["n"], 2)


class ModelTests(unittest.TestCase):
    def training(self):
        return [{"id": "a1", "label": "a", "features": [0, 1]},
                {"id": "a2", "label": "a", "features": [2, 1]},
                {"id": "b1", "label": "b", "features": [8, 1]},
                {"id": "b2", "label": "b", "features": [10, 1]}]

    def test_scaler_uses_training_only(self):
        model = CentroidModel.fit(self.training())
        self.assertEqual(model.means, (5, 1))
        self.assertEqual(model.scales[1], 1)
        model.probabilities([1000, 1])
        self.assertEqual(model.means, (5, 1))

    def test_nearest_centroid_order_and_normalization(self):
        probs = CentroidModel.fit(self.training()).probabilities([0, 1])
        self.assertGreater(probs["a"], probs["b"])
        self.assertAlmostEqual(sum(probs.values()), 1)

    def test_invalid_features(self):
        model = CentroidModel.fit(self.training())
        for values in ([1], [True, 1], [float("nan"), 1], [float("inf"), 1]):
            with self.assertRaises(ValueError): model.probabilities(values)

    def test_training_rejects_duplicate_id(self):
        rows = self.training()
        rows[1]["id"] = rows[0]["id"]
        with self.assertRaises(ValueError): CentroidModel.fit(rows)

    def test_dataset_and_disjoint_experiment_splits(self):
        self.assertEqual(len(load_wine()), 178)
        for seed in SEEDS:
            report = run(seed)
            parts = [set(ids) for ids in report["split_ids"].values()]
            self.assertEqual([len(x) for x in parts], [106, 36, 36])
            self.assertEqual(len(set.union(*parts)), 178)
            self.assertEqual(sum(map(len, parts)), len(set.union(*parts)))

    def test_reproducibility(self):
        self.assertEqual(run(42), run(42))

    def test_experiment_cli(self):
        result = subprocess.run([sys.executable, "experiment.py"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0)
        output = json.loads(result.stdout)
        self.assertEqual(len(output["runs"]), 3)
        self.assertEqual(output["split_counts"]["test"], 36)

    def test_published_metrics_match_recorded_predictions(self):
        saved = json.loads(Path("results.json").read_text())
        self.assertEqual(saved["dataset_sha256"], hashlib.sha256(DATA.read_bytes()).hexdigest())
        for result in saved["runs"]:
            for mode in result["modes"].values():
                predictions = mode["predictions"]
                covered = sum(row["true_label"] in row["prediction_set"] for row in predictions)
                correct = sum(row["true_label"] == row["point_prediction"] for row in predictions)
                self.assertEqual(mode["overall"]["coverage"], covered/len(predictions))
                self.assertEqual(mode["overall"]["point_accuracy"], correct/len(predictions))


if __name__ == "__main__":
    unittest.main()

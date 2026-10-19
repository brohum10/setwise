"""Split-conformal classification sets with optional class-conditional calibration."""

from dataclasses import dataclass
from fractions import Fraction
import math
import re
import statistics


class InvalidEvaluation(ValueError):
    pass


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", value):
        raise InvalidEvaluation("IDs and labels need 1–100 ASCII letters, digits, dots, underscores or hyphens")
    return value


def probability_vector(values, labels):
    if not isinstance(values, dict) or set(values) != set(labels):
        raise InvalidEvaluation("probability keys must match the complete label set")
    for value in values.values():
        if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
            raise InvalidEvaluation("probabilities must be finite numbers in [0, 1]")
    if not math.isclose(math.fsum(values.values()), 1, rel_tol=0, abs_tol=1e-9):
        raise InvalidEvaluation("probabilities must sum to one")
    return {key: values[key] for key in labels}


def checked_rows(rows, labels):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10_000:
        raise InvalidEvaluation("expected 1–10,000 scored examples")
    seen, result = set(), []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "label", "probabilities"}:
            raise InvalidEvaluation("each row needs exactly id, label, probabilities")
        key = identifier(row["id"])
        if key in seen:
            raise InvalidEvaluation("duplicate example ID")
        seen.add(key)
        if row["label"] not in labels:
            raise InvalidEvaluation("unknown true label")
        result.append({**row, "probabilities": probability_vector(row["probabilities"], labels)})
    return sorted(result, key=lambda row: row["id"])


def finite_sample_threshold(scores, alpha):
    """Use the ceil((n+1)*(1-alpha))-th order statistic, not interpolation."""
    if type(alpha) not in (int, float) or not 0 < alpha < 1 or not math.isfinite(alpha):
        raise InvalidEvaluation("alpha must be a finite number strictly between zero and one")
    if not isinstance(scores, (list, tuple)) or any(type(s) not in (int, float) or not 0 <= s <= 1
                                                  or not math.isfinite(s) for s in scores):
        raise InvalidEvaluation("scores must be finite numbers in [0, 1]")
    target = (len(scores)+1)*(1-Fraction(str(alpha)))
    rank = (target.numerator+target.denominator-1)//target.denominator
    return {"n": len(scores), "rank": rank,
            "threshold": sorted(scores)[rank-1] if rank <= len(scores) else None,
            "unbounded": rank > len(scores)}


@dataclass(frozen=True)
class ConformalSets:
    labels: tuple
    alpha: float
    mode: str
    cutoffs: tuple
    calibration_ids: frozenset
    training_ids: frozenset

    def thresholds(self):
        return {key: {"n": n, "rank": rank, "threshold": cutoff, "unbounded": cutoff is None}
                for key, n, rank, cutoff in self.cutoffs}

    def predict(self, example_id, probabilities):
        identifier(example_id)
        if example_id in self.calibration_ids or example_id in self.training_ids:
            raise InvalidEvaluation("evaluation ID overlaps training or calibration data")
        probabilities = probability_vector(probabilities, self.labels)
        thresholds = self.thresholds()
        selected = []
        for label in self.labels:
            entry = thresholds[label if self.mode == "class_conditional" else "__global__"]
            if entry["unbounded"] or 1-probabilities[label] <= entry["threshold"]:
                selected.append(label)
        return tuple(selected)


def calibrate(labels, rows, *, alpha=0.1, mode="global", training_ids=()):
    if not isinstance(labels, (list, tuple)) or not 2 <= len(labels) <= 100:
        raise InvalidEvaluation("expected 2–100 distinct labels")
    for label in labels:
        identifier(label)
    if len(set(labels)) != len(labels):
        raise InvalidEvaluation("duplicate labels")
    labels = tuple(sorted(labels))
    if type(alpha) not in (int, float) or not 0 < alpha < 1 or not math.isfinite(alpha):
        raise InvalidEvaluation("alpha must be a finite number strictly between zero and one")
    if mode not in ("global", "class_conditional"):
        raise InvalidEvaluation("mode must be global or class_conditional")
    training_ids = frozenset(identifier(key) for key in training_ids)
    rows = checked_rows(rows, labels)
    calibration_ids = frozenset(row["id"] for row in rows)
    if calibration_ids & training_ids:
        raise InvalidEvaluation("training and calibration IDs overlap")
    pool_keys = labels if mode == "class_conditional" else ("__global__",)
    pools = {key: [] for key in pool_keys}
    for row in rows:
        key = row["label"] if mode == "class_conditional" else "__global__"
        pools[key].append(1-row["probabilities"][row["label"]])
    entries = {key: finite_sample_threshold(scores, alpha) for key, scores in pools.items()}
    return ConformalSets(labels, alpha, mode,
                         tuple((key, entry["n"], entry["rank"], entry["threshold"]) for key, entry in entries.items()),
                         calibration_ids, training_ids)


def evaluate(model, rows):
    rows = checked_rows(rows, model.labels)
    results = []
    for row in rows:
        selected = model.predict(row["id"], row["probabilities"])
        point = min(model.labels, key=lambda label: (-row["probabilities"][label], label))
        results.append({"id": row["id"], "true_label": row["label"], "point_prediction": point,
                        "prediction_set": list(selected), "covered": row["label"] in selected})

    def metrics(items):
        singleton = [row for row in items if len(row["prediction_set"]) == 1]
        return {"n": len(items), "coverage": statistics.fmean(row["covered"] for row in items),
                "mean_set_size": statistics.fmean(len(row["prediction_set"]) for row in items),
                "point_accuracy": statistics.fmean(row["point_prediction"] == row["true_label"] for row in items),
                "empty_rate": statistics.fmean(not row["prediction_set"] for row in items),
                "singleton_rate": len(singleton)/len(items),
                "singleton_accuracy": statistics.fmean(row["covered"] for row in singleton) if singleton else None}

    return {"overall": metrics(results),
            "by_class": {label: metrics([row for row in results if row["true_label"] == label])
                         for label in model.labels if any(row["true_label"] == label for row in results)},
            "predictions": results}

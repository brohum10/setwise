"""Reproduce held-out Wine experiments; no downloads or external dependencies."""

import csv
import hashlib
import json
from pathlib import Path
import platform
import random

from centroid import CentroidModel
from setwise import calibrate, evaluate


DATA = Path(__file__).with_name("data") / "wine.csv"
SEEDS = (7, 19, 42)


def load_wine(path=DATA):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for index, line in enumerate(csv.reader(handle)):
            if len(line) != 14 or line[0] not in ("1", "2", "3"):
                raise ValueError("expected Wine class + 13 features")
            rows.append({"id": f"wine-{index:03}", "label": line[0], "features": [float(v) for v in line[1:]]})
    if len(rows) != 178:
        raise ValueError("expected 178 Wine observations")
    return rows


def run(seed, rows=None):
    rows = load_wine() if rows is None else rows
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    train = [rows[i] for i in order[:106]]
    calibration = [rows[i] for i in order[106:142]]
    test = [rows[i] for i in order[142:]]
    split_ids = {"train": sorted(r["id"] for r in train),
                 "calibration": sorted(r["id"] for r in calibration), "test": sorted(r["id"] for r in test)}
    assert not (set(split_ids["train"]) & set(split_ids["calibration"]))
    assert not (set(split_ids["test"]) & (set(split_ids["train"]) | set(split_ids["calibration"])))
    model = CentroidModel.fit(train)
    labels = [label for label, _ in model.centers]
    def score(items):
        return [{"id": row["id"], "label": row["label"],
                 "probabilities": model.probabilities(row["features"])} for row in items]
    calibration_scores, test_scores = score(calibration), score(test)
    modes = {}
    for mode in ("global", "class_conditional"):
        conformal = calibrate(labels, calibration_scores, alpha=0.1, mode=mode, training_ids=model.training_ids)
        modes[mode] = {"thresholds": conformal.thresholds(), **evaluate(conformal, test_scores)}
    return {"seed": seed, "split_ids": split_ids, "modes": modes}


def main():
    report = {"dataset": "UCI Wine", "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
              "python": platform.python_version(), "alpha": 0.1,
              "model": "train-only standardized centroid; softmax of negative mean squared distance",
              "split_counts": {"train": 106, "calibration": 36, "test": 36},
              "runs": [run(seed) for seed in SEEDS]}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

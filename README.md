# SetWise

**A classifier can be accurate and still need a way to say “several classes are plausible.”**

[![Tests](https://github.com/brohum10/setwise/actions/workflows/tests.yml/badge.svg)](https://github.com/brohum10/setwise/actions/workflows/tests.yml)

SetWise implements split-conformal classification sets, compares global and class-conditional calibration, and includes a reproducible real-data experiment with a model trained from scratch using Python's standard library.

Python 3.12+ · no pip dependencies · no API keys · bundled public dataset

## Run the complete experiment

Download this repository, open its folder, and run:

```sh
python experiment.py
python -m unittest -v
```

The experiment trains a standardized nearest-centroid model on the public UCI Wine dataset. Each of three fixed seeds uses 106 training, 36 calibration, and 36 test observations. Scaling and class centroids are fitted on training data only. Both conformal methods use the same frozen model and scored examples.

For a reusable local report:

```sh
python experiment.py > local-results.json
```

## Observed results

The following values come from [results.json](results.json), which includes split IDs, thresholds, per-class metrics, and every held-out prediction. Each test split has **36 observations**, so percentages move in increments of about 2.78 points.

| Seed | Point accuracy | Global coverage | Global mean set size | Class-conditional coverage | Class-conditional mean set size |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 97.22% | 100.00% | 1.083 | 97.22% | 1.056 |
| 19 | 97.22% | 86.11% | 0.889 | 86.11% | 1.444 |
| 42 | 100.00% | 91.67% | 0.917 | 94.44% | 1.639 |

All seeds use `alpha = 0.1`, corresponding to nominal 90% coverage under the method's assumptions. **Seed 19 falls below 90% observed coverage.** The experiment retains that result. A statistical coverage statement does not promise that every finite test split reaches the nominal rate.

Mean set size can be below one because empty sets are kept. They signal abstention or incompatibility with the calibration scores; the code does not replace them with a top-one prediction. Larger sets expose ambiguity. Class-conditional sets can also exclude the point classifier's favorite label, so singleton accuracy need not equal point accuracy.

This is a small classical benchmark, not an LLM experiment, a production deployment, or evidence of real-world reliability. The three splits overlap across seeds and must not be treated as 108 independent held-out examples. Hyperparameters are fixed in code; the experiment does not search for the most flattering split.

## How it works

```text
178 public Wine observations
            │ seeded shuffle, disjoint within each run
     ┌──────┼───────────┐
 train 106  │           │
     │      │           │
 fit scaling + centroids│
     └──────┼───────────┤
     calibration 36    test 36
            │           │
      true-label scores │
            │           │
      finite-sample cutoffs
            └───────────┤
                  prediction sets
                        │
           coverage, size, abstention, class breakdown
```

### Baseline model

`centroid.py` fits feature means and population standard deviations, standardizes the training examples, and computes one centroid per class. Constant features use scale one. Scores are softmax-normalized negative mean squared distances to those centroids, with stable exponentiation. These are normalized model scores; they are not claimed to be calibrated class probabilities.

The model is deliberately simple so the calibration behavior stays inspectable. No neural network, pretrained model, hidden API call, or third-party ML package is used.

### Calibration

For each calibration example, the nonconformity score is `1 - p(true_label)`. With n calibration scores, select the order statistic at:

```text
k = ceil((n + 1) * (1 - alpha))
```

Include candidate label y when `1 - p(y) <= cutoff`. The implementation computes the rank with rational arithmetic from the decimal alpha, uses an actual order statistic, and includes ties. It does not interpolate quantiles.

If k exceeds n, the cutoff is unbounded and the corresponding candidate labels are always included. JSON represents this as `threshold: null, unbounded: true`, rather than nonstandard Infinity. Missing calibration classes therefore get conservative inclusion in class-conditional mode.

- **Global:** one cutoff uses all calibration scores. Under exchangeability and a fixed scoring model independent of calibration/test data, split conformal provides a marginal coverage statement.
- **Class-conditional:** one cutoff per true class uses that class's calibration scores. Under the corresponding within-class exchangeability assumptions, the coverage statement is conditional on class. Sparse classes often produce larger or unbounded sets.

Neither statement is a per-example probability that a specific prediction is correct. Distribution shift and adaptive reuse of calibration data can invalidate the assumptions. The implementation follows the simple score-threshold construction discussed in [Angelopoulos and Bates, A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511).

## Use your own classifier scores

```python
from setwise import calibrate, evaluate

calibration = [
    {"id": "cal-1", "label": "cat", "probabilities": {"cat": 0.8, "dog": 0.2}},
    {"id": "cal-2", "label": "dog", "probabilities": {"cat": 0.3, "dog": 0.7}},
]
model = calibrate(["cat", "dog"], calibration, alpha=0.1,
                  mode="global", training_ids=["train-1"])
print(model.thresholds())
print(model.predict("test-1", {"cat": 0.6, "dog": 0.4}))
# This intentionally tiny calibration sample produces an unbounded cutoff.
```

Provide complete probability/normalized-score mappings over the same 2–100 labels. Values must be finite numbers in [0,1] and sum to one within absolute tolerance 1e-9. Booleans are rejected. Calibration/evaluation inputs contain 1–10,000 rows; IDs and labels use 1–100 ASCII letters, digits, dots, underscores, or hyphens.

Calibration and evaluation IDs must be unique within their input. Training IDs cannot overlap calibration, and prediction rejects IDs already used for training/calibration. These checks catch accidental ID reuse, not duplicate content hidden behind different IDs. The caller must freeze the upstream model/scoring procedure and ensure actual dataset separation.

`evaluate(model, test_rows)` returns overall and per-class coverage, mean set size, point accuracy, empty-set rate, singleton rate, singleton accuracy, and individual predictions. No singleton observations means singleton accuracy is JSON `null`. In class breakdowns, a class absent from the test set is omitted rather than assigned a fabricated metric.

## Verification

**28 tests** run on Python 3.12 and 3.14 locally and in GitHub Actions. They check finite-sample ranks, ties, sparse classes, unbounded sets, abstentions, ID overlap, invalid scores, immutability, input-order invariance, train-only scaling, stable model scoring, dataset splits, report metrics, and the real experiment entry point.

An independent leave-one-out rank oracle checks **120 seeded scenarios** with distinct scores: it enumerates which held-out ranks fall inside each calibration threshold. Saved report metrics are also recomputed directly from recorded predictions.

## Data, licensing, and reproducibility

The bundled `data/wine.csv` contains 178 observations and 13 chemical features for three cultivars. Source: Aeberhard, S. & Forina, M. (1992), [Wine, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine), [DOI 10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J).

The dataset is **CC BY 4.0**, separately from the MIT-licensed code. See [DATA_NOTICE.txt](DATA_NOTICE.txt). Only line endings were normalized for the bundled copy. Its SHA-256 is `6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659`.

`results.json` was generated with Python 3.14.2. Seeded splitting, sorted IDs, fixed hyperparameters, and saved split membership make the experiment reproducible. Floating-point last digits can vary across runtimes; the tests run on both supported versions. No external downloads occur when running the experiment.

## Limitations

The Wine dataset is small and comparatively easy. It does not establish robustness under new domains, time-dependent examples, adversarial inputs, or model drift. Class-conditional calibration has very small class-specific samples in this experiment. There is no adaptive threshold tuning, APS/RAPS scoring, deployment monitor, persistent model loader, or production API. IDs and result files provide auditability, not authentication or privacy protection.

## License

Code: MIT, [LICENSE](LICENSE). Dataset: CC BY 4.0, [DATA_NOTICE.txt](DATA_NOTICE.txt).

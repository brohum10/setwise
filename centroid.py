"""A small train-only standardized centroid baseline, with normalized scores."""

from dataclasses import dataclass
import math
import statistics


def vector(values, size=None):
    if not isinstance(values, (list, tuple)) or not values or (size is not None and len(values) != size):
        raise ValueError("incorrect feature dimensions")
    if any(type(v) not in (int, float) or not -1e9 <= v <= 1e9 or not math.isfinite(v) for v in values):
        raise ValueError("features must be finite bounded numbers")
    return tuple(values)


@dataclass(frozen=True)
class CentroidModel:
    means: tuple
    scales: tuple
    centers: tuple
    training_ids: frozenset

    @classmethod
    def fit(cls, rows):
        if not isinstance(rows, list) or len(rows) < 2:
            raise ValueError("need at least two training examples")
        width = len(vector(rows[0]["features"]))
        ids, by_label = set(), {}
        for row in rows:
            vector(row["features"], width)
            if row["id"] in ids:
                raise ValueError("duplicate training ID")
            ids.add(row["id"])
            by_label.setdefault(row["label"], []).append(row["features"])
        if len(by_label) < 2:
            raise ValueError("need at least two training classes")
        columns = list(zip(*(row["features"] for row in rows)))
        means = tuple(statistics.fmean(column) for column in columns)
        scales = tuple(math.sqrt(statistics.pvariance(column)) or 1 for column in columns)
        centers = tuple((label, tuple((statistics.fmean(v[j] for v in values)-means[j])/scales[j]
                                     for j in range(width))) for label, values in sorted(by_label.items()))
        return cls(means, scales, centers, frozenset(ids))

    def probabilities(self, features):
        features = vector(features, len(self.means))
        transformed = tuple((v-m)/s for v, m, s in zip(features, self.means, self.scales))
        distances = {label: math.fsum((a-b)**2 for a, b in zip(transformed, center))/len(features)
                     for label, center in self.centers}
        smallest = min(distances.values())
        weights = {label: math.exp(smallest-distance) for label, distance in distances.items()}
        total = math.fsum(weights.values())
        return {label: weight/total for label, weight in weights.items()}

"""NumPy-only metrics; fixed class universe and group bootstrap."""
import numpy as np


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(shifted)
    return p / p.sum(axis=-1, keepdims=True)


def validate(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if p.ndim != 2 or y.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Expected nonempty labels [N] and probabilities [N,C]")
    if p.shape[1] < 2 or y.min() < 0 or y.max() >= p.shape[1]:
        raise ValueError("Labels outside class universe")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid probabilities")
    return y, p


def confusion(y, pred, nclasses):
    return np.bincount(np.asarray(y) * nclasses + np.asarray(pred),
                       minlength=nclasses ** 2).reshape(nclasses, nclasses)


def classification(y, p, bins=15):
    y, p = validate(y, p)
    pred = p.argmax(1)
    cm = confusion(y, pred, p.shape[1])
    tp, support, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp, dtype=float),
                   where=(support + predicted) > 0)
    conf, correct = p.max(1), pred == y
    assignments = np.minimum((conf * bins).astype(int), bins - 1)
    calibration = []
    ece = 0.
    for b in range(bins):
        mask = assignments == b
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n else None
        confidence = float(conf[mask].mean()) if n else None
        if n:
            ece += n / len(y) * abs(acc - confidence)
        calibration.append(dict(lower=b / bins, upper=(b + 1) / bins, count=n,
                                accuracy=acc, confidence=confidence))
    onehot = np.eye(p.shape[1])[y]
    return dict(n=len(y), accuracy=float(correct.mean()), macro_f1=float(f1.mean()),
                balanced_accuracy=float(recall.mean()), nll=float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()),
                brier=float(((p - onehot) ** 2).sum(1).mean()), ece=float(ece),
                confusion_matrix=cm.tolist(), precision=precision.tolist(), recall=recall.tolist(),
                f1=f1.tolist(), support=support.tolist(), calibration_bins=calibration)


def fit_temperature(logits, y):
    """One scalar chosen on validation data only, bounded to [0.05, 10]."""
    y = np.asarray(y, dtype=int)
    def loss(log_t):
        p = softmax(np.asarray(logits) / np.exp(log_t))
        return -np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()
    lo, hi = np.log(.05), np.log(10.)
    for _ in range(80):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if loss(a) < loss(b):
            hi = b
        else:
            lo = a
    candidate = float(np.exp((lo + hi) / 2))
    return candidate if loss(np.log(candidate)) <= loss(0.) else 1.


def selective(y, p, threshold):
    y, p = validate(y, p)
    keep = p.max(1) >= threshold
    return {"threshold": float(threshold), "coverage": float(keep.mean()), "accepted": int(keep.sum()),
            "accuracy": float((p.argmax(1)[keep] == y[keep]).mean()) if keep.any() else None}


def choose_threshold(y, p, target_accuracy=.85, min_coverage=.25):
    """Validation heuristic, not a guarantee of future or test accuracy."""
    candidates = [selective(y, p, t) for t in np.unique(np.r_[0., p.max(1)])]
    valid = [r for r in candidates if r["coverage"] >= min_coverage and r["accuracy"] >= target_accuracy]
    return max(valid, key=lambda r: r["coverage"]) if valid else {
        "threshold": 0., "coverage": 1., "accuracy": classification(y, p)["accuracy"],
        "target_unmet": True}


def group_bootstrap(y, p, groups, repetitions=1000, seed=2026):
    """Resample source groups, retaining all their clips. CIs assume independent groups."""
    y, p = validate(y, p)
    groups = np.asarray(groups)
    if len(groups) != len(y) or repetitions < 1:
        raise ValueError("Invalid bootstrap inputs")
    unique = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        ix = np.concatenate([indices[g] for g in rng.choice(unique, len(unique), replace=True)])
        cm = confusion(y[ix], p[ix].argmax(1), p.shape[1])
        den = cm.sum(0) + cm.sum(1)
        f1 = np.divide(2 * np.diag(cm), den, out=np.zeros(p.shape[1]), where=den > 0).mean()
        values.append((float((y[ix] == p[ix].argmax(1)).mean()), float(f1)))
    values = np.asarray(values)
    return {name: np.quantile(values[:, i], [.025, .975]).tolist()
            for i, name in enumerate(("accuracy", "macro_f1"))}


def paired_bootstrap(y, a, b, groups, repetitions=1000, seed=2026):
    """Paired video/group bootstrap of macro-F1(A)-macro-F1(B)."""
    y, a = validate(y, a)
    _, b = validate(y, b)
    if a.shape != b.shape or len(groups) != len(y):
        raise ValueError("Paired predictions must align")
    groups = np.asarray(groups)
    unique = np.unique(groups)
    lookup = {g: np.flatnonzero(groups == g) for g in unique}
    rng, differences = np.random.default_rng(seed), []
    for _ in range(repetitions):
        ix = np.concatenate([lookup[g] for g in rng.choice(unique, len(unique), replace=True)])
        differences.append(classification(y[ix], a[ix])["macro_f1"] -
                           classification(y[ix], b[ix])["macro_f1"])
    return {"macro_f1_difference": classification(y, a)["macro_f1"] - classification(y, b)["macro_f1"],
            "ci95": np.quantile(differences, [.025, .975]).tolist(),
            "note": "Within-seed paired test-set uncertainty; not training-seed uncertainty."}


def ema(probabilities, alpha=.5):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or len(p) == 0 or not 0 < alpha <= 1:
        raise ValueError("EMA requires nonempty [T,C] and alpha in (0,1]")
    out = p.copy()
    for i in range(1, len(out)):
        out[i] = alpha * p[i] + (1 - alpha) * out[i - 1]
    return out

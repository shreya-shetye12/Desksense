import json
from pathlib import Path

import numpy as np

from .common import device_for, digest, manifest_rows, write_json, read_json
from .metrics import (choose_threshold, classification, fit_temperature, group_bootstrap,
                      selective, softmax)
from .models import load_model
from .train import loader, predict


def run_predictions(args, split, perturb="none"):
    device = device_for(args.device)
    model, bundle = load_model(args.run + "/best.pt", device)
    if digest(args.manifest) != bundle["manifest_sha256"] and not getattr(args, "external", False):
        raise ValueError("Manifest changed since training; use original, or explicitly --external for a new test set")
    rows = [r for r in manifest_rows(args.manifest) if r["split"] == split]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate evaluation IDs")
    if getattr(args, "external", False):
        original = bundle["source_fingerprints"]
        known_hashes = {r["sha256"] for r in original}
        known_groups = {r["group"] for r in original}
        known_paths = {r["path"] for r in original}
        external_hashes = set()
        for row in rows:
            sha = digest(row["path"])
            if sha in known_hashes or row["group"] in known_groups or row["path"] in known_paths:
                raise ValueError(f"External evaluation overlaps original data: {row['id']}")
            if sha in external_hashes:
                raise ValueError("Exact duplicate content within external test data")
            external_hashes.add(sha)
    classes = bundle["spec"]["classes"]
    if any(r["label"] not in classes for r in rows):
        raise ValueError("Evaluation labels differ from training")
    batches = loader(rows, bundle["spec"], args.batch_size, args.workers,
                     views=args.views, perturb=perturb, seed=2026)
    logits, y, indices = predict(model, batches, device)
    return logits, y, [rows[i] for i in indices], bundle


def calibrate(args):
    logits, y, rows, bundle = run_predictions(args, "val")
    temperature = fit_temperature(logits, y)
    probs = softmax(logits / temperature)
    chosen = choose_threshold(y, probs, args.target_accuracy, args.min_coverage)
    result = dict(temperature=temperature, selection=chosen,
                  target_accuracy=args.target_accuracy, min_coverage=args.min_coverage,
                  raw=classification(y, softmax(logits)), calibrated=classification(y, probs),
                  checkpoint_sha256=digest(Path(args.run) / "best.pt"),
                  manifest_sha256=digest(args.manifest), views=args.views,
                  validation_ids=[r["id"] for r in rows],
                  note="Temperature and abstention threshold fit on validation only. Accuracy target is not guaranteed.")
    write_json(Path(args.run) / "calibration.json", result)
    print(f"Validation temperature={temperature:.3f}, threshold={chosen['threshold']:.3f}")


def evaluate(args):
    run = Path(args.run)
    cal = read_json(run / "calibration.json")
    if cal["checkpoint_sha256"] != digest(run / "best.pt") or cal["views"] != args.views:
        raise ValueError("Checkpoint or evaluation views changed; rerun calibration")
    logits, y, rows, bundle = run_predictions(args, "test", args.perturb)
    raw = softmax(logits)
    probs = softmax(logits / cal["temperature"])
    groups = [r["group"] for r in rows]
    result = dict(split="test", perturbation=args.perturb, external=args.external,
                  synthetic=bundle["config"]["synthetic"], seed=bundle["config"]["seed"],
                  model=bundle["spec"]["model"], mode=bundle["config"]["mode"],
                  checkpoint_sha256=cal["checkpoint_sha256"],
                  training_manifest_sha256=bundle["manifest_sha256"],
                  evaluation_manifest_sha256=digest(args.manifest), classes=bundle["spec"]["classes"],
                  views=args.views, raw=classification(y, raw), calibrated=classification(y, probs),
                  bootstrap_ci95=group_bootstrap(y, probs, groups, args.bootstrap),
                  selective=selective(y, probs, cal["selection"]["threshold"]),
                  risk_coverage=[selective(y, probs, t) for t in np.linspace(0, 1, 101)],
                  perturbation_note="For interventions, original labels are retained. Reversal/shuffling may change "
                                    "or invalidate the action; scores then measure original-label retention.",
                  bootstrap_note="Source-group percentile bootstrap; does not measure training-seed uncertainty.")
    # Optional predeclared domain metadata; never derive subgroups from correctness.
    for field in ("domain", "lighting", "camera"):
        values = sorted({r.get(field, "") for r in rows} - {""})
        if values:
            result[field + "_slices"] = {
                value: classification(y[[r.get(field) == value for r in rows]],
                                      probs[[r.get(field) == value for r in rows]]) for value in values}
    out = Path(args.out) if args.out else run / f"eval_{'external_' if args.external else ''}{args.perturb}"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", result)
    with (out / "predictions.jsonl").open("w") as handle:
        for row, label, scores, z in zip(rows, y, probs, logits):
            handle.write(json.dumps({**row, "target": int(label), "prediction": int(scores.argmax()),
                                     "probabilities": scores.tolist(), "logits": z.tolist()}) + "\n")
    print(f"Test macro-F1={result['calibrated']['macro_f1']:.4f}; artifacts: {out}")

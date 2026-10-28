import base64
import csv
import html
import io
import json
from pathlib import Path

import numpy as np

from .common import read_json, write_json
from .metrics import paired_bootstrap


def predictions(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compare(args):
    a, b = predictions(args.a), predictions(args.b)
    if len({r["id"] for r in a}) != len(a) or len({r["id"] for r in b}) != len(b):
        raise ValueError("Duplicate prediction IDs")
    a, b = sorted(a, key=lambda r: r["id"]), sorted(b, key=lambda r: r["id"])
    ma, mb = read_json(Path(args.a).parent / "metrics.json"), read_json(Path(args.b).parent / "metrics.json")
    for field in ("classes", "evaluation_manifest_sha256", "views", "external", "synthetic"):
        if ma[field] != mb[field]:
            raise ValueError(f"Cannot pair evaluations with different {field}")
    for x, y in zip(a, b):
        if (x["id"], x["target"], x["group"]) != (y["id"], y["target"], y["group"]):
            raise ValueError("Predictions must have identical IDs, targets and groups")
    if len(a) != len(b):
        raise ValueError("Prediction count mismatch")
    result = paired_bootstrap([r["target"] for r in a], [r["probabilities"] for r in a],
                              [r["probabilities"] for r in b], [r["group"] for r in a], args.bootstrap)
    write_json(args.out, {"a": args.a, "b": args.b, **result})
    print(result)


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from textwrap import fill
    root, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(root.rglob("metrics.json"))
    evaluations = [(p, read_json(p)) for p in paths]
    evaluations = [(p, m) for p, m in evaluations if "calibrated" in m]
    if not evaluations:
        raise ValueError("No clip evaluation metrics found")
    synthetic = any(m["synthetic"] for _, m in evaluations)
    warning = "SYNTHETIC SOFTWARE CHECK — NOT MODEL PERFORMANCE" if synthetic else (
        "EXPLORATORY REAL-DATA CPU PILOT — FROZEN BACKBONES, SMALL NONRANDOM SUBSET"
        if any(m.get("pilot", False) for _, m in evaluations) else "Measured held-out clip evaluation")
    sections, summary, errors = [], [], []
    def figure_data(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    for path, m in evaluations:
        name = str(path.parent.relative_to(root))
        stats = m["calibrated"]
        ci = m["bootstrap_ci95"]["macro_f1"]
        summary.append(dict(run=name, model=m["model"], mode=m["mode"], seed=m["seed"],
                            perturbation=m["perturbation"], external=m["external"],
                            macro_f1=stats["macro_f1"], ci_low=ci[0], ci_high=ci[1],
                            accuracy=stats["accuracy"], raw_ece=m["raw"]["ece"], calibrated_ece=stats["ece"],
                            coverage=m["selective"]["coverage"], selective_accuracy=m["selective"]["accuracy"],
                            videos=stats["n"], manifest=m["evaluation_manifest_sha256"]))
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        cm = np.asarray(stats["confusion_matrix"])
        axes[0].imshow(cm, cmap="Blues")
        axes[0].set(xticks=range(len(cm)), yticks=range(len(cm)), xlabel="Predicted class ID",
                    ylabel="True class ID", title="Confusion counts")
        for i in range(len(cm)):
            for j in range(len(cm)):
                axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                             color="white" if cm[i, j] > cm.max() / 2 else "black")
        for label, metric in (("Raw", m["raw"]), ("Calibrated", stats)):
            bins = [b for b in metric["calibration_bins"] if b["count"]]
            axes[1].plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", label=label)
        axes[1].plot([0, 1], [0, 1], "k--", alpha=.5)
        axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy", title="Reliability")
        axes[1].legend()
        points = sorted((r for r in m["risk_coverage"] if r["accuracy"] is not None), key=lambda r: r["coverage"])
        axes[2].plot([r["coverage"] for r in points], [1 - r["accuracy"] for r in points], "o-", markersize=3)
        axes[2].set(xlim=(0, 1), ylim=(0, 1), xlabel="Coverage", ylabel="Error rate", title="Selective prediction")
        fig.suptitle(fill(name, 110))
        image = figure_data(fig)
        classes = "<br>".join(f"{i}: {html.escape(c)} (n={stats['support'][i]})" for i, c in enumerate(m["classes"]))
        rows = predictions(path.parent / "predictions.jsonl")
        wrong = sorted((r for r in rows if r["target"] != r["prediction"]),
                       key=lambda r: max(r["probabilities"]), reverse=True)[:12]
        error_rows = []
        for row in wrong:
            record = dict(run=name, id=row["id"], path=row["path"],
                          true=m["classes"][row["target"]], predicted=m["classes"][row["prediction"]],
                          confidence=max(row["probabilities"]))
            errors.append(record)
            error_rows.append("<tr>" + "".join(f"<td>{html.escape(str(record[k]))}</td>"
                                              for k in ("id", "true", "predicted", "confidence")) + "</tr>")
        sections.append(f"<section><h2>{html.escape(name)}</h2><img alt='Evaluation charts' src='{image}'>"
                        f"<details><summary>Class legend and support</summary>{classes}</details>"
                        "<h3>Highest-confidence mistakes</h3><table><tr><th>ID</th><th>True</th>"
                        f"<th>Predicted</th><th>Confidence</th></tr>{''.join(error_rows)}</table></section>")
    def save_csv(path, records):
        if records:
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    save_csv(out / "summary.csv", summary)
    save_csv(out / "failure_examples.csv", errors)
    headers = ["run", "macro_f1", "ci_low", "ci_high", "accuracy", "raw_ece", "calibrated_ece", "coverage", "videos"]
    table = "<tr>" + "".join(f"<th>{k}</th>" for k in headers) + "</tr>"
    for row in summary:
        table += "<tr>" + "".join(f"<td>{html.escape(f'{row[k]:.4f}' if isinstance(row[k], float) else str(row[k]))}</td>"
                                   for k in headers) + "</tr>"
    # Aggregate clean, same-dataset configurations across independent training seeds only.
    grouped = {}
    for _, m in evaluations:
        if m["perturbation"] != "none":
            continue
        key = (m["model"], m["mode"], m["evaluation_manifest_sha256"], m["external"], m["views"])
        seeds = grouped.setdefault(key, {})
        if m["seed"] in seeds:
            raise ValueError("Duplicate configuration/seed evaluation; narrow --runs directory")
        seeds[m["seed"]] = m["calibrated"]["macro_f1"]
    aggregate = []
    for key, seeds in grouped.items():
        values = list(seeds.values())
        aggregate.append(dict(model=key[0], mode=key[1], manifest=key[2], external=key[3], views=key[4],
                              seeds=sorted(seeds), mean_macro_f1=float(np.mean(values)),
                              sample_std_macro_f1=float(np.std(values, ddof=1)) if len(values) > 1 else None))
    write_json(out / "seed_summary.json", aggregate)
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>DeskSense evaluation</title><style>
body{{font:15px system-ui;margin:32px auto;max-width:1250px;padding:0 18px;background:#f5f7fa;color:#182535}}
h1{{letter-spacing:-1px}}section{{background:white;border-radius:12px;padding:20px;margin:24px 0}}
img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #ddd}}
.banner{{padding:15px;background:#e7edf8;border-radius:8px}}details{{margin:16px 0}}.scroll{{overflow:auto}}
</style><h1>DeskSense · evaluation report</h1><p class="banner">{warning}</p>
<p>Percentile 95% intervals resample source groups. They do not capture training-seed uncertainty.
Calibration and rejection thresholds use validation only. Streaming/EMA results require separate continuous annotations.
Intervention scores retain original labels; reversal/shuffle may change or invalidate actions.
No test-based model selection is performed by this report.</p>
<div class="scroll"><table>{table}</table></div>{''.join(sections)}</html>"""
    (out / "index.html").write_text(page)
    lines = ["# DeskSense measured results", "", warning, "",
             "| Run | Macro-F1 | 95% group CI | Accuracy |", "|---|---:|---|---:|"]
    for row in summary:
        lines.append(f"| {row['run']} | {row['macro_f1']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {row['accuracy']:.4f} |")
    lines += ["", "Do not select the best test seed. Report predeclared configurations and seed mean/std.",
              "See seed_summary.json for training-seed variability and failure_examples.csv for error review."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"Report: {out / 'index.html'}")

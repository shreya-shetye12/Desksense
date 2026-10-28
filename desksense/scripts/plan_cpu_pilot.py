"""Freeze a CPU pilot plan before running either model or inspecting test scores."""

import argparse
from pathlib import Path

from desksense.common import digest, manifest_rows, read_json, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--acquisition", required=True)
parser.add_argument("--out", required=True, help="Results directory; plan must not already exist")
args = parser.parse_args()
target = Path(args.out) / "experiment_plan.json"
if target.exists():
    raise FileExistsError("Plan already exists; use a fresh results directory")
meta = read_json(args.acquisition)
rows = manifest_rows(args.manifest)
write_json(
    target,
    {
        "status": "Plan fixed before any test metrics",
        "dataset": "Something-Something V2, bounded archive-prefix pilot",
        "videos": len(rows),
        "counts": {s: sum(r["split"] == s for r in rows) for s in ["train", "val", "test"]},
        "classes": meta["classes"],
        "acquisition_compressed_bytes": meta["downloaded_compressed_bytes"],
        "source": meta["source"],
        "selection": meta["selection"],
        "split_seed": 2026,
        "manifest_sha256": digest(args.manifest),
        "models": ["frame_mean frozen ResNet18/ImageNet", "videomae frozen Kinetics-400 pretraining"],
        "training": "Cached deterministic whole-clip features; train linear head (+ token-mean LayerNorm for VideoMAE); no augmentation or backbone fine-tuning",
        "seeds": [42, 43, 44],
        "epochs": 100,
        "patience": 20,
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.05,
        "label_smoothing": 0.1,
        "selection_metric": "validation macro-F1",
        "calibration": "temperature and reject threshold on val only",
        "test_diagnostics": ["none", "shuffle", "reverse", "repeat"],
        "bootstrap_repetitions": 1000,
        "limitations": [
            "Small nonrandom archive-prefix subset",
            "No actor IDs; source clips disjoint only",
            "CPU-only frozen-backbone pilot; not final fine-tuned benchmark",
            "Model architectures and pretraining differ",
        ],
    },
)
print(target)

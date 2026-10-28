import csv
import hashlib
import subprocess
import sys
from pathlib import Path

from .common import digest, read_json, write_json


def call(*args):
    subprocess.run([sys.executable, "-m", "desksense.cli", *map(str, args)], check=True)


def audit_command(args):
    from .prepare import audit
    write_json(args.out, audit(args.manifest, read_json(args.classes)))


def suite(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    plan = dict(manifest_sha256=digest(args.manifest), classes=read_json(args.classes),
                seeds=args.seeds, epochs=args.epochs, batch_size=args.batch_size, bootstrap=args.bootstrap,
                configurations=[["frame_mean", "partial"], ["videomae", "head"], ["videomae", "partial"]],
                perturbations=["none", "reverse", "shuffle", "repeat"],
                selection="Predeclared comparisons; best epoch by val macro-F1; never select a seed by test")
    plan_path = root / "experiment_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Existing suite has a different plan; choose a fresh --out")
    write_json(plan_path, plan)
    for seed in args.seeds:
        for model, mode in plan["configurations"]:
            run = root / f"{model}_{mode}_s{seed}"
            complete = run / "complete.json"
            if complete.exists() and args.resume:
                if read_json(complete) != plan:
                    raise ValueError("Completed run plan mismatch")
                continue
            if not (run / "best.pt").exists():
                call("train", "--manifest", args.manifest, "--classes", args.classes,
                     "--model", model, "--mode", mode, "--out", run, "--seed", seed,
                     "--epochs", args.epochs, "--batch-size", args.batch_size, "--device", args.device,
                     "--lr", .001 if mode == "head" else .0001)
            else:
                # An interrupted training run has a best checkpoint but may not have completed.
                # Never silently treat that checkpoint as a completed training experiment.
                if not (run / "training_complete.json").exists():
                    raise ValueError(f"Interrupted training in {run}; move that directory and restart this seed")
                config = read_json(run / "config.json")
                if not args.resume or any(config[k] != v for k, v in {
                    "seed": seed, "model": model, "mode": mode, "epochs": args.epochs,
                    "batch_size": args.batch_size, "manifest_sha256": plan["manifest_sha256"]}.items()):
                    raise ValueError(f"Existing run mismatch or --resume missing: {run}")
            call("calibrate", "--run", run, "--manifest", args.manifest,
                 "--batch-size", args.batch_size, "--device", args.device)
            for perturb in plan["perturbations"]:
                call("evaluate", "--run", run, "--manifest", args.manifest, "--perturb", perturb,
                     "--batch-size", args.batch_size, "--device", args.device, "--bootstrap", args.bootstrap)
            write_json(complete, plan)
        for perturb in ("none", "shuffle", "reverse", "repeat"):
            a = root / f"videomae_partial_s{seed}" / "eval_none" / "predictions.jsonl"
            b = root / f"frame_mean_partial_s{seed}" / "eval_none" / "predictions.jsonl" if perturb == "none" else (
                root / f"videomae_partial_s{seed}" / f"eval_{perturb}" / "predictions.jsonl")
            call("compare", "--a", a, "--b", b, "--bootstrap", args.bootstrap,
                 "--out", root / "comparisons" / f"s{seed}_vs_{perturb}.json")
    call("report", "--runs", root, "--out", root / "report")


def smoke(args):
    import cv2
    import numpy as np
    root = Path(args.out).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Smoke output must be empty; choose another --out")
    root.mkdir(parents=True, exist_ok=True)
    classes = ["synthetic_red", "synthetic_blue"]
    write_json(root / "classes.json", classes)
    rows = []
    for split, n in (("train", 8), ("val", 4), ("test", 4)):
        for label, name in enumerate(classes):
            for i in range(n):
                vid = f"{split}_{label}_{i}"
                path = root / f"{vid}.avi"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (48, 48))
                if not writer.isOpened():
                    raise RuntimeError("MJPG encoder unavailable")
                rng = np.random.default_rng(int(hashlib.sha256(vid.encode()).hexdigest()[:8], 16))
                for t in range(8):
                    frame = rng.integers(0, 25, (48, 48, 3), dtype=np.uint8)
                    frame[:, :, 2 if label == 0 else 0] = 180
                    cv2.rectangle(frame, (t * 3, 12), (t * 3 + 12, 24), (80, 200, 80), -1)
                    writer.write(frame)
                writer.release()
                rows.append(dict(id=vid, path=str(path), label=name, split=split, group=vid))
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run = root / "tiny"
    call("train", "--model", "tiny", "--mode", "full", "--manifest", manifest,
         "--classes", root / "classes.json", "--epochs", 2, "--batch-size", 4,
         "--accumulate", 1, "--lr", .01, "--device", "cpu", "--out", run)
    call("calibrate", "--run", run, "--manifest", manifest, "--device", "cpu")
    for perturb in ("none", "shuffle"):
        call("evaluate", "--run", run, "--manifest", manifest, "--device", "cpu",
             "--perturb", perturb, "--bootstrap", 50)
    call("infer", "--run", run, "--video", rows[-1]["path"], "--device", "cpu",
         "--out", root / "timeline.json")
    call("benchmark", "--run", run, "--video", rows[-1]["path"], "--device", "cpu",
         "--warmup", 1, "--repeats", 3)
    call("compare", "--a", run / "eval_none/predictions.jsonl", "--b", run / "eval_shuffle/predictions.jsonl",
         "--bootstrap", 30, "--out", root / "comparison.json")
    continuous = []
    for split, seed in (("val", 901), ("test", 902)):
        path = root / f"continuous_{split}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (48, 48))
        if not writer.isOpened():
            raise RuntimeError("MJPG encoder unavailable")
        rng = np.random.default_rng(seed)
        for frame_id in range(32):
            frame = rng.integers(0, 40, (48, 48, 3), dtype=np.uint8)
            frame[:, :, 2 if frame_id < 16 else 0] = 190
            writer.write(frame)
        writer.release()
        continuous.append(dict(path=str(path), split=split, group=f"continuous_{split}", segments=[
            dict(start=0., end=2., label=classes[0]), dict(start=2., end=4., label=classes[1])]))
    write_json(root / "continuous.json", continuous)
    call("stream-eval", "--run", run, "--annotations", root / "continuous.json", "--split", "test",
         "--device", "cpu", "--window", .5, "--stride", .25, "--bootstrap", 30,
         "--out", root / "stream_metrics.json")
    # Real API startup and video prediction against the trained tiny checkpoint.
    import os
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from .api import app
    with patch.dict(os.environ, {"DESKSENSE_RUN": str(run), "DESKSENSE_DEVICE": "cpu"}):
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            response = client.post("/predict", content=Path(rows[-1]["path"]).read_bytes(),
                                   headers={"content-type": "application/octet-stream"})
            assert response.status_code == 200, response.text
            assert response.json()["predictions"]
    call("report", "--runs", run, "--out", root / "report")
    write_json(root / "SOFTWARE_TEST_ONLY.json", {"synthetic": True,
               "note": "Exercises code paths; these scores MUST NOT be used as real-world evaluation results."})

"""CPU-friendly frozen-feature pilot. No test labels enter head fitting or calibration.

Example:
  HF_HOME=work/hf-cache python scripts/cpu_pilot.py --manifest /path/manifest.csv \
      --work /path/pilot --results /path/results --model videomae

Run each model with the SAME manifest and a predeclared experiment_plan.json.
"""

import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from desksense.common import digest, manifest_rows, read_json, seed_all, write_json
from desksense.metrics import classification, softmax
from desksense.models import VideoClassifier, make_spec
from desksense.prepare import audit
from desksense.train import loader


def extract(model, rows, spec, cache, perturb="none"):
    cache = Path(cache)
    if cache.exists():
        saved = np.load(cache)
        if saved["manifest_sha256"].item() != digest(ARGS.manifest):
            raise ValueError("Feature cache belongs to different manifest")
        return saved["features"]
    values = []
    batches = loader(rows, spec, 1, views=1, perturb=perturb, seed=2026)
    started = time.perf_counter()
    model.eval()
    for n, (clips, _, _) in enumerate(batches):
        x = clips[:, 0]
        with torch.inference_mode():
            if spec["model"] == "videomae":
                feat = model.network.videomae(pixel_values=x).last_hidden_state.mean(1)
            else:
                b, t, c, h, w = x.shape
                feat = model.network(x.reshape(b * t, c, h, w)).reshape(b, t, -1).mean(1)
            values.append(feat.cpu().numpy().copy())
        if (n + 1) % 50 == 0 or n + 1 == len(rows):
            print(
                f"{spec['model']} {perturb}: {n + 1}/{len(rows)} clips; {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    features = np.concatenate(values)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, features=features, manifest_sha256=digest(ARGS.manifest))
    return features


def fit(model, features, rows, spec, seed, root, audit_result, plan):
    seed_all(seed)
    nclasses = len(spec["classes"])
    if spec["model"] == "videomae":
        linear = nn.Linear(features.shape[1], nclasses)
        nn.init.normal_(linear.weight, std=0.02)
        nn.init.zeros_(linear.bias)
        head = nn.Sequential(nn.LayerNorm(features.shape[1]), linear)
    else:
        head = nn.Linear(features.shape[1], nclasses)
    x = torch.from_numpy(features)
    y = torch.tensor([spec["classes"].index(r["label"]) for r in rows])
    tr = torch.tensor([i for i, r in enumerate(rows) if r["split"] == "train"])
    va = torch.tensor([i for i, r in enumerate(rows) if r["split"] == "val"])
    opt = torch.optim.AdamW(head.parameters(), lr=plan["lr"], weight_decay=plan["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=plan["epochs"])
    rng = torch.Generator().manual_seed(seed)
    best, stale, best_state, best_epoch = -1.0, 0, None, 0
    history = []
    for epoch in range(plan["epochs"]):
        head.train()
        order = tr[torch.randperm(len(tr), generator=rng)]
        total = 0.0
        for ix in order.split(plan["batch_size"]):
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(x[ix]), y[ix], label_smoothing=plan["label_smoothing"])
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(ix)
        head.eval()
        with torch.no_grad():
            val = classification(y[va].numpy(), softmax(head(x[va]).numpy()))
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=total / len(tr),
                val_macro_f1=val["macro_f1"],
                val_accuracy=val["accuracy"],
                lr=opt.param_groups[0]["lr"],
            )
        )
        if val["macro_f1"] > best:
            best, stale, best_state, best_epoch = (
                val["macro_f1"],
                0,
                copy.deepcopy(head.state_dict()),
                epoch + 1,
            )
        else:
            stale += 1
        if (epoch + 1) % 10 == 0:
            print(
                spec["model"],
                "seed",
                seed,
                "epoch",
                epoch + 1,
                "val macro-F1",
                round(val["macro_f1"], 4),
                flush=True,
            )
        schedule.step()
        if stale >= plan["patience"]:
            break
    head.load_state_dict(best_state)
    if spec["model"] == "videomae":
        model.network.fc_norm = head[0]
        model.network.classifier = head[1]
    else:
        model.network.fc = head
    config = dict(
        model=spec["model"],
        mode="head",
        seed=seed,
        synthetic=False,
        manifest_sha256=digest(ARGS.manifest),
        manifest=str(Path(ARGS.manifest).resolve()),
        device="cpu",
        torch_version=str(torch.__version__),
        epochs=plan["epochs"],
        feature_policy="Frozen deterministic full-clip features; no training augmentation",
        trainable_parameters=sum(p.numel() for p in head.parameters()),
        total_parameters=sum(p.numel() for p in model.parameters()),
        pilot=True,
        predeclared_plan=plan,
    )
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", config)
    write_json(root / "spec.json", spec)
    write_json(root / "history.json", history)
    write_json(root / "audit.json", audit_result)
    torch.save(
        dict(
            state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            spec=spec,
            config=config,
            epoch=best_epoch,
            manifest_sha256=digest(ARGS.manifest),
            source_fingerprints=audit_result["source_fingerprints"],
        ),
        root / "best.pt",
    )
    write_json(root / "training_complete.json", dict(best_epoch=best_epoch, best_val_macro_f1=best))
    return head


def evaluate_cached(head, features, rows, root, result_root, perturb, plan):
    """Use identical core evaluator metrics on saved cached logits (no repeated decode)."""
    from desksense.metrics import choose_threshold, fit_temperature, group_bootstrap, selective
    import json

    classes = plan["classes"]
    head.eval()
    with torch.no_grad():
        z = head(torch.from_numpy(features)).numpy()
    labels = np.array([classes.index(r["label"]) for r in rows])
    val = np.array([r["split"] == "val" for r in rows])
    test = np.array([r["split"] == "test" for r in rows])
    if perturb == "none":
        temperature = fit_temperature(z[val], labels[val])
        probs = softmax(z[val] / temperature)
        selection = choose_threshold(labels[val], probs, 0.85, 0.25)
        calibration = dict(
            temperature=temperature,
            selection=selection,
            target_accuracy=0.85,
            min_coverage=0.25,
            raw=classification(labels[val], softmax(z[val])),
            calibrated=classification(labels[val], probs),
            checkpoint_sha256=digest(root / "best.pt"),
            manifest_sha256=digest(ARGS.manifest),
            views=1,
            validation_ids=[r["id"] for r in rows if r["split"] == "val"],
        )
        write_json(root / "calibration.json", calibration)
    else:
        calibration = read_json(root / "calibration.json")
    test_rows = [r for r in rows if r["split"] == "test"]
    z, y = z[test], labels[test]
    raw = softmax(z)
    p = softmax(z / calibration["temperature"])
    config = read_json(root / "config.json")
    metrics = dict(
        split="test",
        perturbation=perturb,
        external=False,
        synthetic=False,
        seed=config["seed"],
        model=config["model"],
        mode="head",
        pilot=True,
        checkpoint_sha256=calibration["checkpoint_sha256"],
        training_manifest_sha256=digest(ARGS.manifest),
        evaluation_manifest_sha256=digest(ARGS.manifest),
        classes=classes,
        views=1,
        raw=classification(y, raw),
        calibrated=classification(y, p),
        bootstrap_ci95=group_bootstrap(y, p, [r["group"] for r in test_rows], plan["bootstrap_repetitions"]),
        selective=selective(y, p, calibration["selection"]["threshold"]),
        risk_coverage=[selective(y, p, t) for t in np.linspace(0, 1, 101)],
        perturbation_note="Original-label retention diagnostic; transformed actions may be semantically invalid",
        limitation="Small nonrandom archive-prefix subset, frozen backbones, clip-disjoint but not verified actor-disjoint",
    )
    target = result_root / f"eval_{perturb}"
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "metrics.json", metrics)
    with (target / "predictions.jsonl").open("w") as f:
        for row, label, logits, probs in zip(test_rows, y, z, p):
            f.write(
                json.dumps(
                    {
                        **row,
                        "target": int(label),
                        "prediction": int(probs.argmax()),
                        "probabilities": probs.tolist(),
                        "logits": logits.tolist(),
                    }
                )
                + "\n"
            )
    print(
        config["model"],
        "seed",
        config["seed"],
        perturb,
        "test macro-F1",
        round(metrics["calibrated"]["macro_f1"], 4),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--model", choices=["videomae", "frame_mean"], required=True)
    ARGS = parser.parse_args()
    plan = read_json(Path(ARGS.results) / "experiment_plan.json")
    if digest(ARGS.manifest) != plan["manifest_sha256"]:
        raise ValueError("Plan manifest mismatch")
    rows = manifest_rows(ARGS.manifest)
    classes = plan["classes"]
    audit_result = audit(ARGS.manifest, classes)
    spec = make_spec(ARGS.model, classes)
    model = VideoClassifier(spec, pretrained=True).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if ARGS.model == "frame_mean":
        model.network.fc = nn.Identity()
    features = extract(model, rows, spec, Path(ARGS.work) / f"{ARGS.model}_features.npz")
    seeds = []
    for seed in plan["seeds"]:
        root = Path(ARGS.work) / "runs" / f"{ARGS.model}_head_s{seed}"
        if (root / "best.pt").exists():
            raise FileExistsError(root)
        head = fit(model, features, rows, spec, seed, root, audit_result, plan)
        result = Path(ARGS.results) / "runs" / root.name
        evaluate_cached(head, features, rows, root, result, "none", plan)
        write_json(result / "config.json", read_json(root / "config.json"))
        write_json(result / "history.json", read_json(root / "history.json"))
        write_json(result / "calibration.json", read_json(root / "calibration.json"))
        seeds.append((seed, head, root, result))
    if ARGS.model == "videomae":
        for perturb in ["shuffle", "reverse", "repeat"]:
            # Transform test only; validation logits/calibration remain clean.
            test_rows = [r for r in rows if r["split"] == "test"]
            perturbed = extract(
                model, test_rows, spec, Path(ARGS.work) / f"videomae_test_{perturb}.npz", perturb
            )
            altered = features.copy()
            altered[[r["split"] == "test" for r in rows]] = perturbed
            for seed, head, root, result in seeds:
                evaluate_cached(head, altered, rows, root, result, perturb, plan)
    print("Completed", ARGS.model, flush=True)

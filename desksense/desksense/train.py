import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import device_for, digest, manifest_rows, read_json, seed_all, write_json
from .metrics import classification, softmax
from .models import VideoClassifier, make_spec
from .prepare import audit
from .video import VideoDataset


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def loader(rows, spec, batch_size, workers=0, training=False, views=1, perturb="none", seed=0):
    data = VideoDataset(rows, spec["classes"], spec, training, views, perturb, seed)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(data, batch_size=batch_size, shuffle=training, num_workers=workers,
                      worker_init_fn=worker_seed, generator=generator, pin_memory=torch.cuda.is_available())


@torch.inference_mode()
def predict(model, batches, device):
    model.eval()
    logits, labels, indices = [], [], []
    for x, y, ix in batches:
        b, v, t, c, h, w = x.shape
        # Average logits across temporal views consistently in validation/test.
        out = model(x.reshape(b * v, t, c, h, w).to(device)).reshape(b, v, -1).mean(1)
        logits.append(out.cpu().numpy())
        labels.extend(y.tolist())
        indices.extend(ix.tolist())
    return np.concatenate(logits), np.asarray(labels), np.asarray(indices)


def train(args):
    seed_all(args.seed)
    classes = read_json(args.classes)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Need at least two unique classes")
    audit_result = audit(args.manifest, classes, hash_files=True)
    rows = manifest_rows(args.manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists():
        raise FileExistsError(f"Existing run: {out}. Choose another --out.")
    spec = make_spec(args.model, classes, args.checkpoint, args.revision)
    device = device_for(args.device)
    model = VideoClassifier(spec, pretrained=args.model != "tiny").to(device)
    parameters = model.trainable(args.mode)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    batches = loader(train_rows, spec, args.batch_size, args.workers, True, seed=args.seed)
    val_batches = loader(val_rows, spec, args.batch_size, args.workers, views=args.views, seed=args.seed)
    config = {k: v for k, v in vars(args).items() if k != "func"}
    config.update(device=str(device), torch_version=str(torch.__version__),
                  trainable_parameters=sum(p.numel() for p in parameters),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  manifest_sha256=digest(args.manifest), synthetic=args.model == "tiny")
    write_json(out / "config.json", config)
    write_json(out / "audit.json", audit_result)
    write_json(out / "spec.json", spec)
    history, best, stale = [], -1., 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total, seen = 0., 0
        for step, (x, y, _) in enumerate(batches):
            x, y = x[:, 0].to(device), y.to(device)
            # Explicit group size handles a final incomplete accumulation group correctly.
            group_start = (step // args.accumulate) * args.accumulate
            group_size = min(args.accumulate, len(batches) - group_start)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            (loss / group_size).backward()
            if (step + 1) % args.accumulate == 0 or step + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_total += loss.item() * len(y)
            seen += len(y)
        logits, labels, _ = predict(model, val_batches, device)
        metrics = classification(labels, softmax(logits))
        record = dict(epoch=epoch + 1, train_loss=loss_total / seen,
                      val_macro_f1=metrics["macro_f1"], val_accuracy=metrics["accuracy"],
                      lr=optimizer.param_groups[0]["lr"])
        history.append(record)
        write_json(out / "history.json", history)
        print(record, flush=True)
        if metrics["macro_f1"] > best:
            best, stale = metrics["macro_f1"], 0
            torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            spec=spec, config=config, epoch=epoch + 1,
                            source_fingerprints=audit_result["source_fingerprints"],
                            manifest_sha256=config["manifest_sha256"]), out / "best.pt")
        else:
            stale += 1
        schedule.step()
        if stale >= args.patience:
            break
    print(f"Best validation macro-F1: {best:.4f}; saved {out / 'best.pt'}")
    write_json(out / "training_complete.json", {"epochs_completed": len(history), "best_val_macro_f1": best})

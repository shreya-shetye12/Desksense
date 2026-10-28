"""Import an authorized local Something-Something V2 copy; no dataset is redistributed."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import digest, manifest_rows, read_json, write_json


def normalize(label):
    return label.replace("[", "").replace("]", "")


def prepare_ssv2(args):
    classes = read_json(args.classes)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Need at least two unique classes")
    root = Path(args.videos).resolve()
    paths = {}
    for suffix in ("*.webm", "*.mp4", "*.avi"):
        for path in root.rglob(suffix):
            if path.stem in paths:
                raise ValueError(f"Ambiguous video ID: {path.stem}")
            paths[path.stem] = path
    rng = np.random.default_rng(args.seed)
    rows = []
    missing = []
    for official, target in ((args.train_json, "train"), (args.val_json, "test")):
        buckets = defaultdict(list)
        for item in read_json(official):
            label = normalize(item["template"])
            if label in classes:
                vid = str(item["id"])
                if vid not in paths:
                    missing.append(vid)
                else:
                    buckets[label].append(vid)
        for label in classes:
            ids = sorted(buckets[label])
            if len(ids) < (10 if target == "train" else 2):
                raise ValueError(f"Too few {target} videos for {label}: {len(ids)}; missing={len(missing)}")
            rng.shuffle(ids)
            if args.max_per_class:
                ids = ids[:args.max_per_class]
            nval = max(2, round(len(ids) * args.val_fraction)) if target == "train" else 0
            if target == "train" and len(ids) - nval < 2:
                raise ValueError("Too few training videos after split")
            for i, vid in enumerate(ids):
                split = "val" if target == "train" and i < nval else target
                rows.append(dict(id=vid, path=str(paths[vid]), label=label, split=split, group=vid))
    if missing and not args.allow_missing:
        raise ValueError(f"{len(missing)} selected-class videos missing. First IDs: {missing[:10]}. "
                         "Fix dataset or explicitly use --allow-missing and disclose the subset.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "path", "label", "split", "group"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["id"]))
    write_json(out.with_suffix(".provenance.json"), {
        "dataset": "Something-Something V2", "classes": classes, "seed": args.seed,
        "official_train_sha256": digest(args.train_json), "official_val_sha256": digest(args.val_json),
        "policy": "Official train stratified into train/val; official validation used as held-out test",
        "missing_ids": missing, "manifest_sha256": digest(out),
        "note": "group=source clip ID; this does NOT guarantee actor-disjoint splits."
    })
    print(f"Wrote {len(rows)} videos to {out}")
    audit(out, classes)


def audit(manifest, classes=None, hash_files=True):
    rows = manifest_rows(manifest)
    if not rows:
        raise ValueError("Empty manifest")
    ids, groups, paths, hashes = set(), {}, {}, {}
    counts = Counter()
    fingerprints = []
    labels = set(classes or [r["label"] for r in rows])
    for row in rows:
        if not row["id"] or row["id"] in ids:
            raise ValueError(f"Empty or duplicate ID: {row['id']}")
        ids.add(row["id"])
        split = row["split"]
        if split not in {"train", "val", "test"} or row["label"] not in labels or not row["group"]:
            raise ValueError(f"Invalid manifest row: {row}")
        for mapping, key, name in ((groups, row["group"], "group"), (paths, row["path"], "path")):
            if key in mapping and mapping[key] != split:
                raise ValueError(f"Cross-split {name} leakage: {key}")
            mapping[key] = split
        if not Path(row["path"]).is_file():
            raise FileNotFoundError(row["path"])
        if hash_files:
            sha = digest(row["path"])
            if sha in hashes:
                raise ValueError(f"Exact duplicate video content: {row['id']} and {hashes[sha]}")
            hashes[sha] = row["id"]
            fingerprints.append({"id": row["id"], "group": row["group"], "path": row["path"],
                                 "split": split, "sha256": sha})
        counts[(split, row["label"])] += 1
    for split in ("train", "val", "test"):
        for label in labels:
            if counts[(split, label)] == 0:
                raise ValueError(f"Missing class {label} in {split}")
    result = {"manifest_sha256": digest(manifest), "videos": len(rows), "groups": len(groups),
              "counts": {f"{s}/{c}": n for (s, c), n in sorted(counts.items())},
              "exact_content_checked": hash_files, "source_fingerprints": fingerprints,
              "limitation": "Cannot detect near-duplicates or shared actors without additional metadata."}
    print(f"Audit passed: {len(rows)} videos, {len(groups)} groups")
    return result

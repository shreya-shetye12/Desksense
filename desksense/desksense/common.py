import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest_rows(path):
    path = Path(path).resolve()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "path", "label", "split", "group"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires {sorted(required)}")
        rows = list(reader)
    for row in rows:
        video = Path(row["path"])
        row["path"] = str(video if video.is_absolute() else (path.parent / video).resolve())
    return rows


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def device_for(requested="auto"):
    import torch
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

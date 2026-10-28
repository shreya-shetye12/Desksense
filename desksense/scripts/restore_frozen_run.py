"""Restore a measured frozen-backbone run from its compact head artifact.

Backbone weights are downloaded from the pinned original pretrained source.
Only use trusted local artifacts; this uses torch's weights-only loader.
"""

import argparse
from pathlib import Path

import torch

from desksense.common import digest, write_json
from desksense.models import VideoClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--head", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
artifact = torch.load(args.head, map_location="cpu", weights_only=True)
root = Path(args.out)
if root.exists() and any(root.iterdir()):
    raise FileExistsError("Choose a fresh output directory")
model = VideoClassifier(artifact["spec"], pretrained=True)
missing, unexpected = model.load_state_dict(artifact["head_state_dict"], strict=False)
if unexpected:
    raise ValueError(unexpected)
expected = {k for k in model.state_dict() if k.startswith(tuple(artifact["head_prefixes"]))}
if set(artifact["head_state_dict"]) != expected:
    raise ValueError("Head parameter keys do not match the architecture")
root.mkdir(parents=True, exist_ok=True)
checkpoint = {k: artifact[k] for k in ["spec", "config", "epoch", "manifest_sha256", "source_fingerprints"]}
checkpoint["state_dict"] = model.state_dict()
torch.save(checkpoint, root / "best.pt")
calibration = artifact["calibration"]
calibration["checkpoint_sha256"] = digest(root / "best.pt")
write_json(root / "calibration.json", calibration)
write_json(root / "spec.json", artifact["spec"])
write_json(root / "config.json", artifact["config"])
write_json(root / "training_complete.json", {"restored_frozen_run": True, "epoch": artifact["epoch"]})
print(root)

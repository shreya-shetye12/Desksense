import io
import zipfile
import numpy as np
import argparse
import csv
import json
import tarfile
import time
from pathlib import Path
from collections import Counter
import requests

parser = argparse.ArgumentParser(description="Acquire a bounded official SSv2 archive-prefix research pilot")
parser.add_argument("--work", required=True)
parser.add_argument("--classes", default="configs/classes.json")
parser.add_argument("--max-compressed-bytes", type=int, default=1_500_000_000)
args = parser.parse_args()
ROOT = Path(args.work).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
CLASSES = json.loads(Path(args.classes).read_text())
# The provider's research-use dataset terms apply; no videos are redistributed here.

base = "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/"
labels = requests.get(base + "20bn-something-something-download-package-labels.zip", timeout=60)
labels.raise_for_status()
with zipfile.ZipFile(io.BytesIO(labels.content)) as archive:
    for name in archive.namelist():
        if name.endswith(".json"):
            (ROOT / Path(name).name).write_bytes(archive.read(name))
records = {}
for file, split in [("train.json", "development"), ("validation.json", "test")]:
    for item in json.loads((ROOT / file).read_text()):
        label = item["template"].replace("[", "").replace("]", "")
        if label in CLASSES:
            records[str(item["id"])] = (label, split)
(ROOT / "videos").mkdir(exist_ok=True)
url = "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-00"
response = requests.get(url, stream=True, timeout=(20, 90))
response.raise_for_status()


class BoundedReader:
    def __init__(self, raw):
        self.raw = raw
        self.bytes = 0

    def read(self, n=-1):
        if self.bytes >= args.max_compressed_bytes:
            raise EOFError("Pilot compressed-byte budget reached")
        chunk = self.raw.read(min(n if n >= 0 else 65536, args.max_compressed_bytes - self.bytes))
        self.bytes += len(chunk)
        return chunk


reader = BoundedReader(response.raw)
counts = Counter()
seen = 0
selected = []
started = time.time()
try:
    with tarfile.open(fileobj=reader, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            seen += 1
            vid = Path(member.name).stem
            if vid in records:
                label, split = records[vid]
                limit = 80 if split == "development" else 15
                if counts[(split, label)] < limit:
                    if member.size > 10_000_000:
                        raise ValueError("Unexpected oversized clip")
                    (ROOT / "videos" / f"{vid}.webm").write_bytes(tar.extractfile(member).read())
                    counts[(split, label)] += 1
                    selected.append(dict(id=vid, label=label, official_partition=split))
            if seen % 1000 == 0:
                print(
                    json.dumps(
                        dict(
                            seen=seen,
                            compressed_MB=round(reader.bytes / 1e6),
                            selected=len(selected),
                            elapsed=round(time.time() - started),
                        )
                    ),
                    flush=True,
                )
            if all(
                counts[(split, label)] >= limit
                for split, limit in [("development", 80), ("test", 15)]
                for label in CLASSES
            ):
                break
except (EOFError, tarfile.ReadError) as exc:
    print("Bounded stream stopped:", str(exc), flush=True)
finally:
    response.close()
result = dict(
    source=url,
    source_partition="00",
    max_compressed_bytes=args.max_compressed_bytes,
    downloaded_compressed_bytes=reader.bytes,
    archive_members_scanned=seen,
    classes=CLASSES,
    selection="First available clips per class/official partition in bounded archive prefix; exploratory, not representative full benchmark",
    counts={f"{split}/{label}": n for (split, label), n in counts.items()},
    selected=selected,
)
(ROOT / "acquisition.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result["counts"], indent=2), flush=True)

# Recreate the same fixed source-disjoint development split.

rng = np.random.default_rng(2026)
rows = []
for label in CLASSES:
    development = sorted(
        [r["id"] for r in selected if r["label"] == label and r["official_partition"] == "development"]
    )
    rng.shuffle(development)
    nval = max(2, round(0.2 * len(development)))
    for i, vid in enumerate(development):
        rows.append(
            dict(
                id=vid,
                path=str(ROOT / "videos" / f"{vid}.webm"),
                label=label,
                split="val" if i < nval else "train",
                group=vid,
            )
        )
    for r in selected:
        if r["label"] == label and r["official_partition"] == "test":
            rows.append(
                dict(
                    id=r["id"],
                    path=str(ROOT / "videos" / f"{r['id']}.webm"),
                    label=label,
                    split="test",
                    group=r["id"],
                )
            )
with (ROOT / "manifest.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "path", "label", "split", "group"])
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: r["id"]))
print("Manifest:", ROOT / "manifest.csv")

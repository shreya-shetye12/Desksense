import hashlib
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@lru_cache(maxsize=16384)
def _metadata_cached(path, size, mtime_ns):
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        # VP9/WebM container frame-count estimates can exceed decodable frames.
        # Count actual frames; cache by path/size/mtime so epochs reuse the result.
        count = 0
        while cap.grab():
            count += 1
        if count < 1 or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video metadata: {path}")
        return count, fps
    finally:
        cap.release()


def metadata(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _metadata_cached(str(path), stat.st_size, stat.st_mtime_ns)


def sample_indices(count, nframes, start=0, end=None):
    end = count if end is None else min(count, end)
    start = max(0, start)
    if nframes < 1 or end <= start:
        raise ValueError("Invalid temporal sampling bounds")
    return np.linspace(start, end - 1, nframes).round().astype(int)


def decode(path, indices, size=224):
    cap = cv2.VideoCapture(str(path))
    frames = {}
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {path}")
        wanted = set(int(i) for i in indices)
        if not wanted or min(wanted) < 0:
            raise ValueError("Invalid frame indices")
        # Sequential decoding avoids inaccurate non-keyframe seeks in WebM/VFR files.
        for index in range(max(wanted) + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Decode failed at frame {index}: {path}")
            if index not in wanted:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            scale = size / min(h, w)
            frame = cv2.resize(frame, (max(size, round(w * scale)), max(size, round(h * scale))))
            h, w = frame.shape[:2]
            top, left = (h - size) // 2, (w - size) // 2
            frames[index] = frame[top:top + size, left:left + size]
        return np.stack([frames[int(i)] for i in indices])
    finally:
        cap.release()


def tensorize(frames, spec):
    arr = frames.astype(np.float32) / 255.
    arr = (arr - np.asarray(spec["mean"], dtype=np.float32)) / np.asarray(spec["std"], dtype=np.float32)
    return torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())


def temporal_perturb(frames, mode, sample_id="", seed=0):
    if mode == "none":
        return frames
    if mode == "reverse":
        return frames[::-1].copy()
    if mode == "repeat":
        return np.repeat(frames[len(frames) // 2:len(frames) // 2 + 1], len(frames), axis=0)
    if mode == "shuffle":
        stable = int(hashlib.sha256(str(sample_id).encode()).hexdigest()[:8], 16)
        return frames[np.random.default_rng(seed + stable).permutation(len(frames))]
    raise ValueError(f"Unknown perturbation: {mode}")


class VideoDataset(Dataset):
    def __init__(self, rows, classes, spec, training=False, views=1, perturb="none", seed=0):
        self.rows, self.spec, self.training = rows, spec, training
        self.mapping = {c: i for i, c in enumerate(classes)}
        self.views, self.perturb, self.seed = views, perturb, seed
        if not rows or views < 1:
            raise ValueError("Dataset must be nonempty with positive views")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        count, _ = metadata(row["path"])
        clips = []
        for view in range(1 if self.training else self.views):
            if self.training:
                span = max(1, round(count * np.random.uniform(.8, 1.)))
                start = np.random.randint(count - span + 1)
            elif self.views == 1:
                span, start = count, 0
            else:
                span = max(1, round(count * .8))
                start = round((count - span) * view / (self.views - 1))
            indices = sample_indices(count, self.spec["frames"], start, start + span)
            frames = decode(row["path"], indices, self.spec["size"])
            # No horizontal flips: left/right action semantics must stay intact.
            if self.training:
                gain = np.random.uniform(.8, 1.2)
                frames = np.clip(frames.astype(float) * gain, 0, 255).astype(np.uint8)
            frames = temporal_perturb(frames, self.perturb, row["id"], self.seed)
            clips.append(tensorize(frames, self.spec))
        return torch.stack(clips), self.mapping[row["label"]], index

import platform
import time
from pathlib import Path

import numpy as np
import torch

from .common import device_for, digest, read_json, sync, write_json
from .metrics import ema, softmax
from .models import load_model
from .video import decode, metadata, sample_indices, tensorize


class Predictor:
    def __init__(self, run, device="auto"):
        self.device = device_for(device)
        self.model, bundle = load_model(str(Path(run) / "best.pt"), self.device)
        self.spec = bundle["spec"]
        self.source_fingerprints = bundle["source_fingerprints"]
        self.classes = self.spec["classes"]
        self.calibration = read_json(Path(run) / "calibration.json")
        if self.calibration["checkpoint_sha256"] != digest(Path(run) / "best.pt"):
            raise ValueError("Stale calibration")

    @torch.inference_mode()
    def logits(self, x):
        return self.model(x.to(self.device)).cpu().numpy()

    def sliding(self, path, window=2., stride=.5, alpha=.5, max_seconds=600.):
        if window <= 0 or stride <= 0 or stride > window or not 0 < alpha <= 1:
            raise ValueError("Require window>0, 0<stride<=window and 0<alpha<=1")
        count, fps = metadata(path)
        duration = count / fps
        if duration > max_seconds:
            raise ValueError(f"Video exceeds {max_seconds:g}s limit")
        width, step = max(1, round(window * fps)), max(1, round(stride * fps))
        # No future frames: emit each window only at its end. Include a short video as one window.
        ends = list(range(min(width, count), count + 1, step))
        if ends[-1] != count:
            ends.append(count)
        raw, windows = [], []
        for end in ends:
            start = max(0, end - width)
            frames = decode(path, sample_indices(count, self.spec["frames"], start, end), self.spec["size"])
            z = self.logits(tensorize(frames, self.spec).unsqueeze(0))[0]
            raw.append(softmax(z / self.calibration["temperature"]))
            windows.append((start / fps, end / fps))
        smooth = ema(raw, alpha)
        threshold = self.calibration["selection"]["threshold"]
        output = []
        for bounds, p, q in zip(windows, raw, smooth):
            output.append(dict(start=round(bounds[0], 4), end=round(bounds[1], 4),
                               available_at=round(bounds[1], 4),
                               raw_label=self.classes[int(np.argmax(p))], raw_confidence=float(max(p)),
                               raw_probabilities=np.asarray(p).tolist(),
                               label=self.classes[int(q.argmax())], confidence=float(q.max()),
                               abstain=bool(q.max() < threshold), probabilities=q.tolist()))
        return dict(classes=self.classes, duration=duration, window=window, stride=stride, alpha=alpha,
                    threshold=threshold, predictions=output,
                    warning="Calibration/threshold validated on whole clips, not streaming windows or EMA. "
                            "Streaming confidence is a heuristic; validate on annotated continuous video.")


def infer(args):
    predictor = Predictor(args.run, args.device)
    result = predictor.sliding(args.video, args.window, args.stride, args.alpha)
    write_json(args.out, result)
    print(f"Saved {len(result['predictions'])} windows to {args.out}")


def benchmark(args):
    predictor = Predictor(args.run, args.device)
    count, _ = metadata(args.video)
    ix = sample_indices(count, predictor.spec["frames"])
    def prepare():
        frames = decode(args.video, ix, predictor.spec["size"])
        return tensorize(frames, predictor.spec).unsqueeze(0)
    x = prepare().to(predictor.device)
    @torch.inference_mode()
    def forward():
        return predictor.model(x)
    for _ in range(args.warmup):
        forward()
    sync(predictor.device)
    model_times, e2e_times = [], []
    for _ in range(args.repeats):
        sync(predictor.device)
        start = time.perf_counter()
        forward()
        sync(predictor.device)
        model_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        z = predictor.logits(prepare())
        softmax(z / predictor.calibration["temperature"])
        sync(predictor.device)
        e2e_times.append(time.perf_counter() - start)
    def summarize(times):
        return dict(p50_ms=float(np.median(times) * 1000),
                    p95_ms=float(np.quantile(times, .95) * 1000),
                    clips_per_second=float(1 / np.mean(times)))
    result = dict(batch_size=1, frames=predictor.spec["frames"], size=predictor.spec["size"],
                  model=summarize(model_times), decode_preprocess_transfer_forward=summarize(e2e_times),
                  device=str(predictor.device), platform=platform.platform(),
                  hardware=torch.cuda.get_device_name(predictor.device) if predictor.device.type == "cuda" else platform.machine(),
                  torch_version=str(torch.__version__), warmup=args.warmup, repeats=args.repeats,
                  video_sha256=digest(args.video), checkpoint_sha256=digest(Path(args.run) / "best.pt"),
                  note="Warm process, repeated single video, OS/model caches warm; excludes upload and startup. "
                       "clips/sec is not source video FPS or a multi-user throughput measurement.")
    write_json(args.out or Path(args.run) / "benchmark.json", result)
    print(result)

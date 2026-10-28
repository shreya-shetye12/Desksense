"""Evaluate causal smoothing only on genuinely annotated continuous videos."""
import numpy as np

from .common import read_json, write_json
from .inference import Predictor
from .metrics import classification, group_bootstrap


def stream_evaluate(args):
    predictor = Predictor(args.run, args.device)
    videos = read_json(args.annotations)
    from .common import digest
    known_hashes = {r["sha256"] for r in predictor.source_fingerprints}
    group_splits, seen_paths, content_splits = {}, set(), {}
    for video in videos:
        if video["split"] not in {"val", "test"} or not video["group"]:
            raise ValueError("Continuous evaluation requires val/test splits and nonempty groups")
        if video["path"] in seen_paths:
            raise ValueError("Duplicate continuous source video path")
        seen_paths.add(video["path"])
        sha = digest(video["path"])
        if sha in known_hashes:
            raise ValueError("Continuous evaluation overlaps original clip dataset")
        if sha in content_splits:
            raise ValueError("Exact duplicate continuous source content")
        content_splits[sha] = video["split"]
        group = video["group"]
        if group in group_splits and group_splits[group] != video["split"]:
            raise ValueError("Continuous source-group leakage between validation and test")
        group_splits[group] = video["split"]
    all_y, raw_p, smooth_p, groups = [], [], [], []
    details = []
    for video in videos:
        if video["split"] != args.split:
            continue
        segments = video["segments"]
        previous_end = -1.
        for s in segments:
            if not 0 <= s["start"] < s["end"] or s["start"] < previous_end:
                raise ValueError("Segments must be sorted, nonoverlapping, positive intervals")
            if s["label"] not in predictor.classes:
                raise ValueError(s["label"])
            previous_end = s["end"]
        result = predictor.sliding(video["path"], args.window, args.stride, args.alpha)
        if segments and segments[-1]["end"] > result["duration"] + .05:
            raise ValueError("Annotation extends beyond video duration")
        raw, smooth, y, times = [], [], [], []
        for p in result["predictions"]:
            # Last frame in [start,end): use end-epsilon for label lookup.
            time = p["end"] - 1e-6
            matches = [s for s in segments if s["start"] <= time < s["end"]]
            if not matches:
                continue
            y.append(predictor.classes.index(matches[0]["label"]))
            raw.append(p["raw_probabilities"])
            smooth.append(p["probabilities"])
            times.append(p["available_at"])
        if not y:
            continue
        transitions = []
        for s in segments[1:]:
            candidates = [i for i, t in enumerate(times) if s["start"] < t <= s["end"]]
            target = predictor.classes.index(s["label"])
            def delay(probs):
                correct = [i for i in candidates if np.argmax(probs[i]) == target]
                return times[correct[0]] - s["start"] if correct else None
            transitions.append(dict(start=s["start"], label=s["label"], raw_delay=delay(raw),
                                    smoothed_delay=delay(smooth)))
        all_y.extend(y)
        raw_p.extend(raw)
        smooth_p.extend(smooth)
        groups.extend([video["group"]] * len(y))
        # Count adjacent emitted labels on full timeline, including unlabeled intervals.
        raw_labels = [p["raw_label"] for p in result["predictions"]]
        smooth_labels = [p["label"] for p in result["predictions"]]
        def flips(labels):
            return sum(a != b for a, b in zip(labels, labels[1:])) / max(result["duration"] / 60, 1e-6)
        details.append(dict(path=video["path"], group=video["group"], transitions=transitions,
                            raw_flips_per_min=flips(raw_labels), smooth_flips_per_min=flips(smooth_labels)))
    if not all_y:
        raise ValueError("No labeled windows for requested split")
    raw_p, smooth_p = np.asarray(raw_p), np.asarray(smooth_p)
    def delay_summary(key):
        values = [t[key] for v in details for t in v["transitions"]]
        detected = [x for x in values if x is not None]
        return dict(transitions=len(values), missed=sum(v is None for v in values),
                    median_detected_delay=float(np.median(detected)) if detected else None)
    write_json(args.out, dict(split=args.split, alpha=args.alpha, stride=args.stride, window=args.window,
                             raw=classification(all_y, raw_p), smoothed=classification(all_y, smooth_p),
                             raw_ci95=group_bootstrap(all_y, raw_p, groups, args.bootstrap),
                             smoothed_ci95=group_bootstrap(all_y, smooth_p, groups, args.bootstrap),
                             raw_delays=delay_summary("raw_delay"), smooth_delays=delay_summary("smoothed_delay"),
                             videos=details, note="Tune alpha/window/stride on continuous validation only. "
                             "Delay is first correct emitted label; missed transitions reported separately. "
                             "Clip temperature is not validated for streaming confidence."))
    print(f"Saved continuous-video evaluation to {args.out}")

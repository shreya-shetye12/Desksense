# DeskSense

**Can a video Transformer distinguish an actual interaction from a visually similar pretend action?**

DeskSense is an eight-class video interaction recognition project built around that question. It includes VideoMAE transfer learning, an order-independent frame baseline, split-leakage checks, temporal interventions, validation-only confidence calibration, group-bootstrap confidence intervals, multi-seed experiments, continuous-video smoothing evaluation, a FastAPI service, and Docker deployment.

**Real performance is not prefilled.** Run the experiments to produce `metrics.json`, prediction files, `RESULTS.md`, CSV summaries, and a self-contained HTML report. Synthetic smoke scores exercise the software and are explicitly marked as unsuitable for a resume.

## Local video dashboard

Run your trained model with `DESKSENSE_RUN=runs/conda_demo DESKSENSE_DEVICE=cpu uvicorn desksense.api:app --host 127.0.0.1 --port 8000`, then open **http://localhost:8000/**. Upload and play a video, inspect synchronized predictions and confidence traces, compare raw versus smoothed scores, and export the JSON. No separate frontend installation is required. See [docs/UI.md](docs/UI.md) for setup, controls and interpretation.

## 1. Install

Python 3.10–3.12 recommended. Use an NVIDIA GPU notebook or workstation for VideoMAE training. CPU works for tests, the tiny smoke model and inference; full VideoMAE training is expensive. MPS is supported on a best-effort basis.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# On CUDA machines install the matching torch/torchvision build from pytorch.org first.
pip install -e '.[dev]'
python -m unittest discover -s tests -v
ruff check desksense tests
python -m desksense.cli smoke --out runs/synthetic_smoke
```

The install uses bounded compatible release families, rather than pretending a portable CUDA lockfile exists. Save your resolved environment with `pip freeze > runs/environment.txt` after choosing the appropriate PyTorch build. Training artifacts also record the PyTorch version, model configuration, resolved Hub revision where available, seed, manifest hash, and hyperparameters.

The executable `desksense` and `python -m desksense.cli` are equivalent.

## 2. Dataset

Obtain Something-Something V2 through its current authorized distribution. You need the local videos, `something-something-v2-train.json`, and `something-something-v2-validation.json`. This repository neither redistributes dataset videos nor assumes a particular third-party mirror is authorized. Dataset acquisition and model downloads require internet access; training/evaluation can then use the local files. Links and provenance notes are in [docs/DATA.md](docs/DATA.md).

Default class pairs in `configs/classes.json`:

- Picking something up / pretending to pick it up
- Putting something on a surface / pretending to put it on a surface
- Opening something / pretending to open it
- Closing something / pretending to close it

```bash
python -m desksense.cli prepare \
  --videos /absolute/path/to/20bn-something-something-v2 \
  --train-json /absolute/path/to/something-something-v2-train.json \
  --val-json /absolute/path/to/something-something-v2-validation.json \
  --out data/manifest.csv
```

The official training set is stratified into train and validation. The official labeled validation set becomes this project's **held-out test**. This is a custom eight-class task, not the official 174-class benchmark or its hidden test set. Source IDs stay in one split. Exact duplicate file contents are rejected. Actor-disjointness and near-duplicate detection are not claimed.

For a separate pilot, add `--max-per-class 150 --out data/pilot.csv`. This caps each official source partition before the internal train/val split. Keep pilot and final runs in separate directories. `--allow-missing` is explicit and records missing IDs; use the complete selected dataset for final results when possible. Rerun prepare with the same arguments for reproducible splits.

Custom datasets use CSV columns `id,path,label,split,group` (optional `domain,lighting,camera`). Relative video paths resolve relative to the CSV. Use the same group for every clip from one recording; if actor IDs are known, split by actor externally before importing the CSV. Every class must appear in all three splits. See `docs/DATA.md`.

## 3. First real run

Run one small pilot first to verify dataset and hardware, rather than launching the full suite immediately.

```bash
python -m desksense.cli train --manifest data/pilot.csv \
  --model videomae --mode head --epochs 2 --out runs/pilot
python -m desksense.cli calibrate --run runs/pilot --manifest data/pilot.csv
python -m desksense.cli evaluate --run runs/pilot --manifest data/pilot.csv
python -m desksense.cli report --runs runs/pilot --out runs/pilot_report
```

Training uses 16 uniformly sampled frames, temporally consistent brightness jitter, a random 80–100% temporal span, and center crops. Horizontal flips are deliberately omitted. Evaluation uses the full clip with deterministic uniform sampling by default. `--views 3` uses three overlapping 80% temporal spans and averages logits; use the same value for training validation, calibration, and evaluation. Center cropping can remove peripheral interactions: review failures rather than assuming preprocessing is harmless.

The VideoMAE backbone starts from `MCG-NJU/videomae-base`, self-supervised on Kinetics-400, **not a checkpoint already supervised on Something-Something V2**. Fine-tuning uses mean-pooled tokens, a trainable normalization layer, and a linear head. Partial tuning unfreezes the final two encoder blocks plus normalization and the classifier. Head-only tuning freezes the backbone. The ResNet-18 baseline averages per-frame logits across the same 16 sampled frames; it cannot represent frame order. Its ImageNet pretraining differs from VideoMAE's, so the architecture comparison alone does not isolate temporal reasoning.

Defaults: AdamW, cosine LR, label smoothing 0.1, gradient clipping, accumulation, and early stopping by validation macro-F1. Use `--batch-size 1 --accumulate 8` if memory is tight. Partial fine-tuning still runs the whole encoder. No promised GPU memory size or training duration is implied.

## 4. Reproducible final experiments

Freeze your class subset, split, and configuration before looking at final test results.

```bash
python -m desksense.cli suite --manifest data/manifest.csv \
  --out runs/final --seeds 42 43 44 --epochs 15 --device cuda
```

This runs three configurations on each seed:

1. ResNet-18 frame-mean baseline, partial tuning.
2. VideoMAE, frozen encoder / trainable head.
3. VideoMAE, final-two-block partial tuning.

Each configuration gets validation calibration and clean/reversed/shuffled/repeated-frame test diagnostics. The suite also writes paired bootstrap differences for VideoMAE partial versus the frame baseline and versus its own interventions. It saves its plan before running. `--resume` skips completed experiments and continues evaluation of completed training runs. It does not pretend to resume optimizer state: an interrupted training run must be moved aside and restarted.

Calibration learns one temperature on validation logits. The optional reject threshold is chosen on validation at a target accuracy and minimum coverage; failure to meet the target is recorded, and no test guarantee is made. Macro-F1 is the primary metric; accuracy, per-class precision/recall, ECE, NLL, multiclass Brier score, risk/coverage, and group-bootstrap 95% intervals are also saved. Positive temperature scaling cannot change argmax accuracy. Three seeds measure training variability separately from bootstrap uncertainty.

**Temporal interventions are diagnostics.** Reversal can change the true action, shuffling can make it invalid, and repeated frames remove motion. Scores retain the original labels and therefore measure original-label retention under intervention, not necessarily accuracy on semantically valid transformed videos. A performance drop is not by itself proof of learned causal reasoning.

```bash
python -m desksense.cli report --runs runs/final --out runs/final_report
```

Open `runs/final_report/index.html`; it embeds its charts and works offline. Inspect `summary.csv`, `seed_summary.json`, `failure_examples.csv`, and the paired comparisons. Reports show all configurations without selecting the best test seed. `failure_examples.csv` points to original videos for human review.

## 5. Streaming inference and latency

```bash
python -m desksense.cli infer --run runs/final/videomae_partial_s42 \
  --video /path/to/video.mp4 --window 2 --stride .5 --alpha .5 \
  --out runs/timeline.json
python -m desksense.cli benchmark --run runs/final/videomae_partial_s42 \
  --video /path/to/video.mp4 --warmup 5 --repeats 30
```

The inference timeline emits overlapping windows only at their end and uses causal EMA; it never consumes future windows. The first prediction requires up to one full window. Short videos produce one window, and the final partial stride is included. Predictions include timestamps, full probabilities, labels and a heuristic abstention flag.

**Whole-clip calibration does not establish calibrated streaming/EMA confidence.** The inference output carries that limitation. This classifier has no unknown/background class and does not detect individual objects or people. It classifies an entire window into known interaction labels; arbitrary webcam footage can be out of distribution.

Benchmarking records batch-one p50/p95 synchronized model-forward latency and separate decode/preprocess/transfer/forward latency, with hardware and video/checkpoint hashes. Warmups are excluded; runs repeat the same video with warm caches. Clip throughput is not source FPS, HTTP latency, or multi-user load capacity. Compare speed only using the same hardware, precision, dimensions, frames and input policy.

## 6. Evaluate smoothing properly

Collect continuous desk recordings with multiple actions, label intervals, and assign entire source/actor groups to validation or test. See `docs/continuous.example.json` for the schema. Replace its paths and labels with real annotations; it contains no actual dataset.

```bash
python -m desksense.cli stream-eval --run runs/final/videomae_partial_s42 \
  --annotations data/continuous.json --split val --alpha .5 --out runs/stream_val.json
# After choosing alpha/window/stride on continuous validation only, freeze them:
python -m desksense.cli stream-eval --run runs/final/videomae_partial_s42 \
  --annotations data/continuous.json --split test --alpha .5 --out runs/stream_test.json
```

Reports include raw versus smoothed window metrics, group-bootstrap intervals, label flips/minute, first-correct transition delay, and missed transitions. Ground truth is the action at the end of each emitted window. Unannotated gaps are excluded from accuracy. Delay is conditional on successful detection; missed counts must be reported alongside it. Lower flicker alone is not better performance. Use novel desk recordings as a distinct domain-shift evaluation, not additional training after inspecting their test mistakes.

## 7. FastAPI and Docker

```bash
DESKSENSE_RUN=runs/final/videomae_partial_s42 uvicorn desksense.api:app --port 8000
curl http://localhost:8000/health
curl -X POST 'http://localhost:8000/predict?window=2&stride=0.5&alpha=0.5' \
  -H 'Content-Type: application/octet-stream' --data-binary @/path/to/video.mp4
```

```bash
docker build -t desksense .
docker run --rm -p 8000:8000 \
  -v "$PWD/runs/final/videomae_partial_s42:/models/run:ro" desksense
```

Inference reloads the architecture from the saved configuration; it does not download backbone weights. The CPU image is a local/demo deployment. The API bounds uploads to 100 MiB and decoded duration to 120 seconds, serializes model access, returns 429 when busy, and cleans temporary files. A public service needs authentication, quotas and process-isolated decoding; this is not a hardened untrusted-media service.

## Repository

```text
desksense/
  prepare.py       dataset importer and leakage audit
  video.py         deterministic sampling / decoding / perturbations
  models.py        VideoMAE, frame-mean baseline, CI-only tiny model
  train.py         transfer learning, validation selection, checkpoints
  metrics.py       metrics, calibration, group/paired bootstrap, EMA
  evaluate.py      validation calibration and held-out evaluation
  stream_eval.py   continuous-video stability / detection delay
  inference.py     timestamped inference and synchronized benchmarks
  report.py        offline HTML report, CSV, Markdown and seed summaries
  api.py           bounded streaming upload and inference endpoint
  workflows.py     fixed experiment suite and synthetic integration check
  cli.py           all command-line entry points
configs/classes.json
tests/              metric, leakage, model and API tests
notebooks/          Colab starter
```

See `docs/EVALUATION.md` for the experiment contract and `docs/RESUME.md` for evidence-based resume wording. No real-world accuracy is asserted by this source release.

## Verified source release

See [docs/VERIFICATION.md](docs/VERIFICATION.md) for completed checks and limits. The included [software-check report](docs/software-check-report.html) is generated from synthetic fixtures and demonstrates the output format; it contains no real-dataset performance evidence.

For a smaller real-data experiment on a CPU, see [docs/CPU_PILOT.md](docs/CPU_PILOT.md). It includes bounded official-data acquisition and frozen-feature head training, with its own predeclared protocol and limitations.

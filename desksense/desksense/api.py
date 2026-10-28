"""Run with DESKSENSE_RUN=runs/video_s42 uvicorn desksense.api:app."""
import json
import secrets
import time
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .inference import Predictor


@asynccontextmanager
async def lifespan(app):
    run = os.environ.get("DESKSENSE_RUN")
    if not run:
        raise RuntimeError("Set DESKSENSE_RUN to a trained and calibrated run directory")
    app.state.predictor = Predictor(run, os.environ.get("DESKSENSE_DEVICE", "auto"))
    app.state.lock = threading.Lock()
    app.state.exports = {}
    yield
    app.state.exports.clear()


app = FastAPI(title="DeskSense", version="0.1.0", lifespan=lifespan)
WEB_ROOT = Path(__file__).resolve().parent / "web"
app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(WEB_ROOT / "index.html", headers={"Cache-Control": "no-cache"})


def configured_sample():
    """Serve only a sample explicitly configured by the local server operator."""
    setting = os.environ.get("DESKSENSE_SAMPLE_VIDEO")
    if not setting:
        return None
    path = Path(setting).expanduser().resolve()
    return path if path.is_file() else None


@app.get("/sample-video", include_in_schema=False)
def sample_video():
    path = configured_sample()
    if path is None:
        raise HTTPException(404, "No sample video configured")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


def expire_exports():
    entries = app.state.exports
    cutoff = time.monotonic() - 15 * 60
    for token, entry in list(entries.items()):
        if entry["created"] < cutoff:
            del entries[token]
    while len(entries) > 8:
        del entries[next(iter(entries))]


@app.get("/exports/{token}.json", include_in_schema=False)
async def export_predictions(token: str):
    expire_exports()
    entry = app.state.exports.get(token)
    if entry is None:
        raise HTTPException(404, "Export expired or unavailable. Analyze the video again.")
    return Response(json.dumps(entry["payload"], indent=2, allow_nan=False) + "\n",
                    media_type="application/json",
                    headers={"Content-Disposition": 'attachment; filename="desksense-predictions.json"',
                             "Cache-Control": "no-store"})


@app.get("/health")
def health():
    predictor = app.state.predictor
    kind = getattr(predictor, "spec", {}).get("model", "unknown")
    name = {"videomae": "VideoMAE", "frame_mean": "Frame-mean ResNet-18", "tiny": "Tiny test model"}.get(kind, "Video classifier")
    return {"status": "ready", "classes": predictor.classes,
            "device": str(predictor.device), "model": kind, "model_name": name,
            "sample_available": configured_sample() is not None}


@app.post("/predict")
async def predict(request: Request, window: float = Query(2., gt=0, le=30),
                  stride: float = Query(.5, gt=0, le=30), alpha: float = Query(.5, gt=0, le=1)):
    """POST raw video bytes with Content-Type: application/octet-stream.

    Streaming request body is bounded before decoding; no multipart spool allocation.
    """
    if stride > window:
        raise HTTPException(422, "stride must be <= window")
    limit = 100 * 1024 * 1024
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > limit:
                raise HTTPException(413, "Maximum upload: 100 MiB")
        except ValueError:
            raise HTTPException(400, "Invalid content-length")
    if not app.state.lock.acquire(blocking=False):
        raise HTTPException(429, "Inference worker busy; retry later")
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".video", delete=False) as handle:
            path = Path(handle.name)
            total = 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > limit:
                    raise HTTPException(413, "Maximum upload: 100 MiB")
                handle.write(chunk)
        if not total:
            raise HTTPException(400, "Empty video")
        result = await run_in_threadpool(app.state.predictor.sliding, path, window, stride, alpha, 120.)
        token = secrets.token_urlsafe(24)
        app.state.exports[token] = {"created": time.monotonic(), "payload": result}
        expire_exports()
        return {**result, "download_url": f"/exports/{token}.json"}
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        if path is not None:
            path.unlink(missing_ok=True)
        app.state.lock.release()

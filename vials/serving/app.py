from contextlib import asynccontextmanager
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

from fastapi import FastAPI, Query, Request, Response

from vials.config import (
    DRIFT_DEFAULT_DAYS,
    INFERENCE_LOG_PATH,
    MODEL_PATH,
    UPSTREAM_HOST,
    UPSTREAM_PORT,
)
from vials.monitoring.drift import detect_drift, load_baseline_stats, load_inference_logs

scoring_proc = None


@asynccontextmanager
async def lifespan(app):
    global scoring_proc
    scoring_proc = _start_scoring_server
    yield
    scoring_proc.terminate()
    scoring_proc.wait(timeout=10)


app = FastAPI(lifespan=lifespan, docs_url=None, openapi_url=None, redoc_url=None)


def _start_scoring_server():
    env = os.environ.copy()
    # The scoring server execs `uvicorn` through PATH; make sure the
    # interpreter's bin directory is on it
    interpreter_bin = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = os.pathsep.join([interpreter_bin, env.get("PATH", "")])
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "serve",
            "-m",
            MODEL_PATH,
            "-p",
            str(UPSTREAM_PORT),
            "--workers",
            "1",
            "--host",
            UPSTREAM_HOST,
            "--no-conda",
        ],
        env=env,
    )
    return proc


def _upstream_health():
    """Return the scoring server's /health status code, None if unreachable."""
    try:
        with urllib.request.urlopen(
            f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}/health",
            timeout=1.0,
        ) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


@app.get("/health")
def health():
    if scoring_proc is not None and scoring_proc.poll() is not None:
        return Response(
            content='{"status": "error"}', status_code=503, media_type="application/json"
        )
    if _upstream_health() == 200:
        return Response(content='{"status": "ok"}', media_type="application/json")
    return Response(
        content='{"status": "loading"}', status_code=503, media_type="application/json"
    )


@app.get("/version")
def version():
    """Model version of the served model (watched by the monitoring platform)."""
    version_file = MODEL_PATH / "version.json"
    if version_file.exists():
        with open(version_file) as f:
            return json.load(f)
    return {"model_version": "unknown", "trained_at": None}


@app.post("/invocations")
@app.post("/infer")
async def invocations(request: Request):
    body = await request.body()
    req = urllib.request.Request(
        f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}/invocations",
        data=body,
        headers={"Content-Type": request.headers.get("content-type", "application/json")},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return Response(
                content=resp.read(),
                status_code=resp.status,
                media_type=resp.headers.get("content-type", "application/json"),
            )
    except urllib.error.HTTPError as e:
        return Response(
            content=e.read(),
            status_code=e.code,
            media_type=e.headers.get("content-type", "application/json"),
        )
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return Response(
            content=f'{{"status": "error", "detail": "{e}"}}',
            status_code=502,
            media_type="application/json",
        )


@app.get("/drift")
def drift(days: int = Query(default=DRIFT_DEFAULT_DAYS, ge=1)):
    """Run drift detection over the requests logged by the scoring server."""
    logs = load_inference_logs(INFERENCE_LOG_PATH)
    baseline = load_baseline_stats()
    return detect_drift(logs, days=days, baseline_stats=baseline)

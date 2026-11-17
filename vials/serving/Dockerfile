# Build from the repo root:  docker build -f vials/serving/Dockerfile -t model-serve .
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only torch, pinned to the versions the model was trained with
RUN pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
COPY vials/serving/serve_requirements.txt .
RUN pip install -r serve_requirements.txt

COPY pyproject.toml README.md ./
COPY vials/ vials/
RUN pip install --no-deps -e .

# Produced by training + export; gitignored, so they must exist before the build
COPY models/model/ models/model/
COPY artifacts/drift_baseline_stats.json artifacts/

EXPOSE 5001
CMD ["uvicorn", "vials.serving.app:app", "--host", "0.0.0.0", "--port", "5001"]

# Vial defect detection

Group project for ITU MLOps course.

Made by Gabija, Marius and Philip.

## What it is

We trained a CNN that takes an image of a pharmaceutical vial and predicts if it is **good** or **defective**.

The pipeline is written in Go with Dagger (`go/pipeline.go`). It runs the tests, downloads the data, trains the model and builds a Docker image with the model in it. GitHub Actions runs the pipeline on every PR, and when something is merged to `main` it pushes the image to ghcr.io and deploys it to Azure.

## Project structure

```
vials/               Python package: data, features, training, export, monitoring, serving
tests/               pytest unit tests for the Python code
go/                  Dagger pipeline in Go and its unit tests
.github/workflows/   GitHub Actions CI/CD workflow
infra/azure/         Azure Container Apps spec
data/                dataset (pulled with DVC)
models/              exported model (created by a run, not committed)
reports/             baseline metrics and results from earlier runs
notebooks/           the original training notebook
```

What each module does:
- `vials/config.py` – all paths and settings in one place
- `vials/dataset.py` – `dvc pull`, download images, stratified 70/15/15 split
- `vials/features.py` – image transforms (resize 64x64, normalise, augment)
- `vials/modeling/models.py` – `SimpleCNN`
- `vials/modeling/train.py` – training run, threshold tuning, MLflow logging and registration
- `vials/modeling/eval.py` – test-set metrics
- `vials/modeling/wrapper.py` – MLflow pyfunc used for serving, logs every request
- `vials/modeling/export.py` – picks the best run (test F1), promotes it, exports to `models/model/`
- `vials/monitoring/drift.py` – drift baseline and drift checks
- `vials/serving/app.py` + `Dockerfile` – FastAPI app (`/health`, `/version`, `/invocations`, `/drift`) in the serving image
- `go/pipeline.go` – Dagger pipeline; `go/helpers.go` + `helpers_test.go` – stage helpers and their Go tests
- `.github/workflows/pipeline.yml` – CI/CD workflow; `infra/azure/app.yaml` – Azure Container Apps spec

## How the model artifact is produced

On every PR and every push to `main`, GitHub Actions first runs the Go unit tests (`go test`) and then the Dagger pipeline: test → data → train → export → model check → serving image. The model check compares the new test F1 with `reports/baseline_metrics.json` and fails the run if it is more than 0.05 worse.

The pipeline exports the model to `output/`, and the workflow uploads that folder as an artifact named `model`. You can download it from the run's page on GitHub, under "Artifacts". It contains:
- `models/model/` – the exported MLflow model and `version.json`
- `models/vial_cnn_model.pth` – the trained PyTorch weights
- `artifacts/` – `model_results.json`, `drift_baseline_stats.json`, `class_names.json`, `image_statistics.json`

On `main`, the serving image is also pushed to `ghcr.io` and deployed to Azure.

## How to use it

The model is running on Azure:

```
https://mlops-itu-jtk.nicetree-802a957f.norwayeast.azurecontainerapps.io
```

Endpoints:
- `GET /health` – check if the model is ready
- `GET /version` – which model version is running
- `POST /invocations` – send an image (base64) and get a prediction
- `GET /drift` – drift report based on the requests so far


## Run it locally

```bash
cd go
go run .
```

This trains the model and builds the image. The model and its metrics are written to `output/`.

To run the Go unit tests:

```bash
cd go && go test -v ./...
```

To run the API locally:

```bash
docker build -f vials/serving/Dockerfile -t model-serve:dev .
docker run --rm -p 5001:5001 model-serve:dev
```

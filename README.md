# Vial defect detection

Group project for ITU MLOps course.

Made by Gabija, Marius and Philip.

## What it is

We trained a CNN that takes an image of a pharmaceutical vial and predicts if it is **good** or **defective**.

The pipeline is written in Go with Dagger (`go/pipeline.go`). It runs the tests, downloads the data, trains the model and builds a Docker image with the model in it. GitHub Actions runs the pipeline on every PR, and when something is merged to `main` it pushes the image to ghcr.io and deploys it to Azure.

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

This trains the model and builds the image. To run the API locally:

```bash
docker build -f vials/serving/Dockerfile -t model-serve:dev .
docker run --rm -p 5001:5001 model-serve:dev
```

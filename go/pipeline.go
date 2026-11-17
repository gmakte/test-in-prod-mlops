package main

import (
	"context"
	"fmt"
	"os"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()
	if err := Build(ctx); err != nil {
		fmt.Println("Pipeline failed:", err)
		os.Exit(1)
	}
}

func Build(ctx context.Context) error {
	// starting Dagger, which then creates and controls docker containers
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stderr))
	if err != nil {
		return err
	}
	defer client.Close()

	// take the repo from root (".."), skipping big local folders in exclude
	src := client.Host().Directory("..", dagger.HostDirectoryOpts{
		Exclude: []string{".venv", "mlflow-state", "artifacts", "models", "data/raw", "go"},
	})

	// a clean Python 3.13 container with our code and libraries installed
	python := client.Container().
		From("python:3.13-slim").
		WithMountedCache("/root/.cache/pip", client.CacheVolume("pip-cache")). //keeps a pip download cache between runs, so the second run is much faster
		WithDirectory("/app", src).                                            //copies repo into the container at /app
		WithWorkdir("/app").
		WithExec([]string{"pip", "install", "-r", "requirements.txt"})

	// TEST stage: lint + unit tests
	python = python.
		WithExec([]string{"ruff", "check"}).
		WithExec([]string{"python", "-m", "pytest", "tests"})

	// run it
	_, err = python.Sync(ctx)
	return err
}

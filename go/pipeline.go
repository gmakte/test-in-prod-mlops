package main

import (
	"context"
	"fmt"
	"os"
	"slices"
	"strings"

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
	// which stages to run, e.g. STAGES="test,data"; empty means all stages
	stages, err := ParseStages(os.Getenv("STAGES"))
	if err != nil {
		return err
	}

	// starting Dagger, which then creates and controls docker containers
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stderr))
	if err != nil {
		return err
	}
	defer client.Close()

	// take the repo from root (".."), skipping big local folders in exclude
	src := client.Host().Directory("..", dagger.HostDirectoryOpts{
		Exclude: []string{".venv", "mlflow-state", "artifacts", "models", "output", "go", "data/raw/**/*.jpg"},
	})

	// a clean Python 3.13 container with our code and libraries installed
	python := client.Container().
		From("python:3.13-slim").
		WithMountedCache("/root/.cache/pip", client.CacheVolume("pip-cache")). //keeps a pip download cache between runs, so the second run is much faster
		WithDirectory("/app", src).                                            //copies repo into the container at /app
		WithWorkdir("/app").
		WithExec([]string{"pip", "install", "-r", "requirements.txt"})

	// run the selected stages (TEST, DATA, TRAIN, EXPORT) in order
	for _, stage := range stages {
		cmds, err := StageCommands(stage)
		if err != nil {
			return err
		}
		for _, cmd := range cmds {
			python = python.WithExec(cmd)
		}
	}

	// without export there is no model to copy out, so just run the stages and stop
	if !slices.Contains(stages, "export") {
		_, err := python.Sync(ctx)
		return err
	}

	// copy the results out of the container to output/
	if _, err := python.Directory("/app/models").Export(ctx, "../output/models"); err != nil {
		return err
	}
	if _, err := python.Directory("/app/artifacts").Export(ctx, "../output/artifacts"); err != nil {
		return err
	}

	// building serving image with Dockerfile
	serveImage := python.Directory("/app").DockerBuild(dagger.DirectoryDockerBuildOpts{
		Dockerfile: "vials/serving/Dockerfile",
	})
	if _, err := serveImage.Sync(ctx); err != nil {
		return err
	}

	if err := Publish(ctx, client, serveImage); err != nil {
		return err
	}

	fmt.Println("Pipeline finished: model exported to output/, serving image built")
	return nil
}

func Publish(ctx context.Context, client *dagger.Client, image *dagger.Container) error {
	token := os.Getenv("GITHUB_TOKEN")
	ref := os.Getenv("GITHUB_REF")
	sha := os.Getenv("GITHUB_SHA")
	owner := strings.ToLower(os.Getenv("GITHUB_REPOSITORY_OWNER"))

	if token == "" || !ShouldPublish(ref) {
		fmt.Println("Skipping publish")
		return nil
	}

	secret := client.SetSecret("github-token", token)
	address := fmt.Sprintf("ghcr.io/%s/model-serve:%s", owner, sha)

	published, err := image.
		WithRegistryAuth("ghcr.io", owner, secret).
		Publish(ctx, address)
	if err != nil {
		return err
	}

	fmt.Println("Published:", published)
	return nil
}

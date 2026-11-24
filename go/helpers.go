package main

import (
	"fmt"
	"strings"
)

// AllStages lists every pipeline stage in the order it runs.
var AllStages = []string{"test", "data", "train", "export"}

// ShouldPublish says whether this branch may publish a Docker image.
// Only the main branch is allowed to.
func ShouldPublish(ref string) bool {
	return ref == "refs/heads/main"
}

// StageCommands returns the commands a stage runs inside the Python container.
func StageCommands(stage string) ([][]string, error) {
	switch stage {
	case "test":
		return [][]string{
			{"ruff", "check"},
			{"python", "-m", "pytest", "tests"},
		}, nil
	case "data":
		return [][]string{{"python", "-m", "vials.dataset"}}, nil
	case "train":
		return [][]string{{"python", "-m", "vials.modeling.train"}}, nil
	case "export":
		return [][]string{{"python", "-m", "vials.modeling.export"}}, nil
	default:
		return nil, fmt.Errorf("unknown stage %q", stage)
	}
}

// ParseStages turns a comma-separated list like "test, data" into stage names.
// An empty or whitespace-only input means all stages.
func ParseStages(input string) ([]string, error) {
	if strings.TrimSpace(input) == "" {
		return AllStages, nil
	}
	var stages []string
	for _, part := range strings.Split(input, ",") {
		name := strings.TrimSpace(part)
		if _, err := StageCommands(name); err != nil {
			return nil, err
		}
		stages = append(stages, name)
	}
	return stages, nil
}

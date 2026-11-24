package main

import (
	"fmt"
	"strings"
)

var AllStages = []string{"test", "data", "train", "export"}

func ShouldPublish(ref string) bool {
	return ref == "refs/heads/main"
}

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

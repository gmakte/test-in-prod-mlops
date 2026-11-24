package main

import (
	"encoding/json"
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

// ParseF1 reads the top level f1_score number from a metrics JSON file
func ParseF1(data []byte) (float64, error) {
	var metrics map[string]any
	if err := json.Unmarshal(data, &metrics); err != nil {
		return 0, fmt.Errorf("invalid metrics JSON: %w", err)
	}
	value, found := metrics["f1_score"]
	if !found {
		return 0, fmt.Errorf("metrics JSON has no \"f1_score\"")
	}
	f1, ok := value.(float64)
	if !ok {
		return 0, fmt.Errorf("\"f1_score\" is not a number: %v", value)
	}
	return f1, nil
}

// CheckF1 fails if the new F1 is not a valid score or is worse than the baseline
func CheckF1(newF1, baselineF1, tolerance float64) error {
	if newF1 < 0 || newF1 > 1 {
		return fmt.Errorf("test F1 %.3f is outside [0, 1] (baseline %.3f)", newF1, baselineF1)
	}
	if newF1 < baselineF1-tolerance {
		return fmt.Errorf("test F1 %.3f is below baseline %.3f minus tolerance %.3f", newF1, baselineF1, tolerance)
	}
	return nil
}

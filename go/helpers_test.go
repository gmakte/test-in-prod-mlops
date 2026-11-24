package main

import (
	"reflect"
	"testing"
)

func TestShouldPublish(t *testing.T) {
	testCases := []struct {
		name   string
		ref    string
		expect bool
	}{
		{"main branch publishes", "refs/heads/main", true},
		{"feature branch does not publish", "refs/heads/feat/dagger", false},
		{"pull request does not publish", "refs/pull/12/merge", false},
		{"empty ref does not publish", "", false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := ShouldPublish(tc.ref)
			if got != tc.expect {
				t.Errorf("ShouldPublish(%q) = %v, want %v", tc.ref, got, tc.expect)
			}
		})
	}
}

func TestStageCommands(t *testing.T) {
	testCases := []struct {
		name      string
		stage     string
		expect    [][]string
		expectErr bool
	}{
		{"test runs ruff and pytest", "test", [][]string{{"ruff", "check"}, {"python", "-m", "pytest", "tests"}}, false},
		{"data runs dataset", "data", [][]string{{"python", "-m", "vials.dataset"}}, false},
		{"train runs training", "train", [][]string{{"python", "-m", "vials.modeling.train"}}, false},
		{"export runs export", "export", [][]string{{"python", "-m", "vials.modeling.export"}}, false},
		{"unknown stage is an error", "deploy", nil, true},
		{"empty stage is an error", "", nil, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := StageCommands(tc.stage)
			if tc.expectErr {
				if err == nil {
					t.Errorf("StageCommands(%q) expected an error, got nil", tc.stage)
				}
				return
			}
			if err != nil {
				t.Fatalf("StageCommands(%q) unexpected error: %v", tc.stage, err)
			}
			if !reflect.DeepEqual(got, tc.expect) {
				t.Errorf("StageCommands(%q) = %v, want %v", tc.stage, got, tc.expect)
			}
		})
	}
}

func TestParseStages(t *testing.T) {
	testCases := []struct {
		name      string
		input     string
		expect    []string
		expectErr bool
	}{
		{"empty input means all stages", "", AllStages, false},
		{"single stage", "test", []string{"test"}, false},
		{"two stages with a space", "test, data", []string{"test", "data"}, false},
		{"unknown stage in list is an error", "test,deploy", nil, true},
		{"typo is an error", "tset", nil, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseStages(tc.input)
			if tc.expectErr {
				if err == nil {
					t.Errorf("ParseStages(%q) expected an error, got nil", tc.input)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseStages(%q) unexpected error: %v", tc.input, err)
			}
			if !reflect.DeepEqual(got, tc.expect) {
				t.Errorf("ParseStages(%q) = %v, want %v", tc.input, got, tc.expect)
			}
		})
	}
}

func TestParseF1(t *testing.T) {
	testCases := []struct {
		name      string
		input     string
		expect    float64
		expectErr bool
	}{
		{"valid JSON returns f1_score", `{"accuracy": 0.79, "f1_score": 0.88}`, 0.88, false},
		{"invalid JSON is an error", `{"f1_score": 0.88`, 0, true},
		{"missing f1_score is an error", `{"accuracy": 0.79}`, 0, true},
		{"string f1_score is an error", `{"f1_score": "high"}`, 0, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseF1([]byte(tc.input))
			if tc.expectErr {
				if err == nil {
					t.Errorf("ParseF1(%q) expected an error, got nil", tc.input)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseF1(%q) unexpected error: %v", tc.input, err)
			}
			if got != tc.expect {
				t.Errorf("ParseF1(%q) = %v, want %v", tc.input, got, tc.expect)
			}
		})
	}
}

func TestCheckF1(t *testing.T) {
	testCases := []struct {
		name      string
		newF1     float64
		baseline  float64
		tolerance float64
		expectErr bool
	}{
		{"equal to baseline passes", 0.88, 0.88, 0.05, false},
		{"within tolerance passes", 0.86, 0.88, 0.05, false},
		{"better than baseline passes", 0.92, 0.88, 0.05, false},
		{"far below baseline fails", 0.70, 0.88, 0.05, true},
		{"negative F1 fails", -0.1, 0.88, 0.05, true},
		{"F1 above 1 fails", 1.2, 0.88, 0.05, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := CheckF1(tc.newF1, tc.baseline, tc.tolerance)
			if tc.expectErr {
				if err == nil {
					t.Errorf("CheckF1(%v, %v, %v) expected an error, got nil", tc.newF1, tc.baseline, tc.tolerance)
				}
				return
			}
			if err != nil {
				t.Errorf("CheckF1(%v, %v, %v) unexpected error: %v", tc.newF1, tc.baseline, tc.tolerance, err)
			}
		})
	}
}

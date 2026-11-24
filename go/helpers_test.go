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

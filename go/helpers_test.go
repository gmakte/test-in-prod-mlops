package main

import "testing"

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

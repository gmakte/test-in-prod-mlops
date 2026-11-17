package main

// ShouldPublish says whether this branch may publish a Docker image.
// Only the main branch is allowed to.
func ShouldPublish(ref string) bool {
	return ref == "refs/heads/main"
}

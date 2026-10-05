# ADR 0002: Upstream PersonaPlex Repository Pinning

## Context
The upstream repository `_personaplex_upstream` was originally committed as a nested unpinned git directory. This created submodule collisions and unrepeatable builds across clean checkouts.

## Decision
We added `_personaplex_upstream` to `.gitignore` and authored `scripts/fetch_upstream.py` which clones the upstream repository at exact pinned commit:
`3428dfd95309a7f3c84fd93259ded0f810d1ff91`

## Consequences
- **Positive:** Deterministic CI and developer clones without submodule conflicts.
- **Positive:** Upstream delimiters (`<system> {prompt} <system>`) and server arguments remain pinned to verified source code.
- **Trade-off:** Running GPU inference locally requires running `python scripts/fetch_upstream.py` once.

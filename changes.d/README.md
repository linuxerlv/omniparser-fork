# News fragments

This directory holds [towncrier](https://towncrier.readthedocs.io/) news
fragments. Each pull request that affects user-visible behavior should add
one file here describing the change. At release time, `towncrier build`
collects every fragment in this directory, formats them by category, and
inserts the rendered section into the top-level `CHANGELOG.md` under a new
`## [<version>] - <date>` heading (above the existing `## [Unreleased]`
marker).

## File naming

Each fragment is a single Markdown file named `<id>.<type>.md`, where:

- `<id>` is the GitHub issue or PR number that the change references. If
  there is no associated issue or PR, prefix the slug with `+` to mark
  the fragment as orphan; towncrier then skips the issue-link rendering
  (e.g. `+dependabot-checkout-v6.misc.md`).
- `<type>` is one of the five categories declared in `pyproject.toml`:

  | Type      | Heading       | Use for                                                                 |
  | --------- | ------------- | ----------------------------------------------------------------------- |
  | `feature` | Added         | New user-visible functionality.                                         |
  | `bugfix`  | Fixed         | Bug fixes affecting user-visible behavior.                              |
  | `doc`     | Documentation | Documentation-only changes that users should know about.                |
  | `removal` | Removed       | Removals, deprecations, breaking changes.                               |
  | `misc`    | Misc          | Internal changes worth noting but with no per-fragment text. Body ignored.|

## Examples

```
123.feature.md                   # PR #123, new feature, gets a #123 link
456.bugfix.md                    # PR #456, bug fix, gets a #456 link
+dependabot-checkout-v6.misc.md  # no PR number; + prefix suppresses link
```

## Content style

Write the fragment as one or more sentences in past tense, ending with a
period. The body is rendered verbatim under the category heading; do not
prepend a bullet (`-`) — towncrier does that for you.

Good:
```
Added `--no-progress` flag to `omniparser-eval` to silence per-row output
during long evaluation runs. Useful when redirecting stdout to a log file.
```

Bad:
```
- adds --no-progress flag        # leading bullet duplicates towncrier's
                                 # rendering and breaks Markdown
```

## Local preview

```bash
uv run --no-sync towncrier build --draft --version $(uv version --package omniparser-core --short)
```

`--draft` writes the rendered Markdown to stdout without modifying
`CHANGELOG.md`. Drop `--draft` to produce the real release entry; this
also deletes the consumed fragment files via `git rm`.

## Configuration

The towncrier config block lives in the root `pyproject.toml` under
`[tool.towncrier]`. The shared workspace `version` field is bumped by
`uv version --package <name> X.Y.Z` (which updates each member's
`pyproject.toml`); update `[tool.towncrier].version` at release time to
match before running `towncrier build`.

# ascend-quant-toolkit 0.4.2 PyPI release record (2026-10-13)

First PyPI release of the offline Ascend Quant Toolkit, published as its own
immutable distribution under the `toolkit-v<version>` tag namespace. The
TestPyPI rehearsal ran on 2026-10-12; the GitHub Release was published on
2026-10-13. This records the exact bytes the index serves; it is not a
performance or accuracy claim.

## Object under test

- Source revision: `50ba3f8` on `main`,
  `ascend_quant_toolkit.__version__ == "0.4.2"` (the distribution version is
  derived from it, so tag and installed metadata cannot drift).
- Tag: `toolkit-v0.4.2`, annotated tag object
  `be9393676a0808733302559f6e513672c08dfa1e`, peeled commit `50ba3f8fdad986fa647b0d55918d13b47a25ec42`.
- GitHub Release: id `411366927`, published
  2026-10-13T06:43:03Z, `draft=False`,
  `prerelease=False`
  (https://github.com/jxd1111/Ascend-LLM-quant/releases/tag/toolkit-v0.4.2).
- Publisher: `.github/workflows/publish-toolkit.yml` on `release: published`,
  OIDC Trusted Publishing through the `pypi` environment; no token is stored.

## TestPyPI rehearsal

- Workflow `rehearse-toolkit-testpypi.yml`, run
  [38222501967](https://github.com/jxd1111/Ascend-LLM-quant/actions/runs/38222501967), run_number 1,
  event `workflow_dispatch`, `head_sha 50ba3f8`,
  conclusion **success**
  (2026-10-12T08:15:49Z to 2026-10-12T08:16:37Z).
- Rehearsed artifacts as served by TestPyPI
  (`https://test.pypi.org/pypi/ascend-quant-toolkit/json`):

| file | size (bytes) | sha256 | uploaded (UTC) |
|---|---|---|---|
| `ascend_quant_toolkit-0.4.2-py3-none-any.whl` | 41003 | `ac8db384383be44404e3ecf8ce6f866b0fa8fc8f745d9a5755cbac4dda291206` | 2026-10-12T08:16:28.187102Z |
| `ascend_quant_toolkit-0.4.2.tar.gz` | 78166 | `cc993bbe43f390489010fe4da7755579bd887b9d346f29d8d425502c8dd3cb32` | 2026-10-12T08:16:29.328914Z |

- Independent local re-verification of those bytes (not on the CI runner): both
  files were re-downloaded from `test-files.pythonhosted.org` and their SHA256
  matched the index records, then the wheel was installed from the TestPyPI
  index into a clean virtual environment with `--no-deps`
  (`python -m pip install --no-deps --index-url https://test.pypi.org/simple/
  ascend-quant-toolkit==0.4.2`); `ascend-quant-toolkit --help` and
  `ascend-quant-toolkit list-recipes` both exited `0` and `list-recipes` printed
  `qwen25-w8a8`. No `torch` distribution is present in that environment, which
  is the point of the torch-free attachment reader: metadata reports an empty
  `Requires:` set and the console script still runs.

## Published artifacts

- Workflow `Publish Toolkit`, run
  [38358286519](https://github.com/jxd1111/Ascend-LLM-quant/actions/runs/38358286519), run_number 1,
  `head_sha 50ba3f8`, conclusion **success**.

- `Build and verify distributions` (attempt 1): completed / **success**
  - 1. Set up job: completed / success
  - 2. Check out the released revision: completed / success
  - 3. Set up Python: completed / success
  - 4. Install build and verification dependencies: completed / success
  - 5. Install both local distributions: completed / success
  - 6. Verify the release tag matches the package version: completed / success
  - 7. Run offline Toolkit tests: completed / success
  - 8. Lint Python sources: completed / success
  - 9. Build the Toolkit wheel and sdist: completed / success
  - 10. Check distribution metadata: completed / success
  - 11. Verify the distributions carry the Toolkit only: completed / success
  - 12. Install the built wheel in an isolated environment: completed / success
  - 13. Upload verified distributions: completed / success
  - 25. Post Set up Python: completed / success
  - 26. Post Check out the released revision: completed / success
  - 27. Complete job: completed / success
- `Publish distributions to PyPI` (attempt 1): completed / **failure**
- `Publish distributions to PyPI` (attempt 2): completed / **success**
  - 1. Set up job: completed / success
  - 2. Download verified distributions: completed / success
  - 3. Publish distributions with PyPI Trusted Publishing: completed / success
  - 6. Post Publish distributions with PyPI Trusted Publishing: completed / success
  - 7. Complete job: completed / success
- `Build and verify distributions` (attempt 2): completed / **success**
  - 1. Set up job: completed / success
  - 2. Check out the released revision: completed / success
  - 3. Set up Python: completed / success
  - 4. Install build and verification dependencies: completed / success
  - 5. Install both local distributions: completed / success
  - 6. Verify the release tag matches the package version: completed / success
  - 7. Run offline Toolkit tests: completed / success
  - 8. Lint Python sources: completed / success
  - 9. Build the Toolkit wheel and sdist: completed / success
  - 10. Check distribution metadata: completed / success
  - 11. Verify the distributions carry the Toolkit only: completed / success
  - 12. Install the built wheel in an isolated environment: completed / success
  - 13. Upload verified distributions: completed / success
  - 25. Post Set up Python: completed / success
  - 26. Post Check out the released revision: completed / success
  - 27. Complete job: completed / success

- Published artifacts as served by PyPI
  (`https://pypi.org/pypi/ascend-quant-toolkit/json`):

| file | size (bytes) | sha256 | uploaded (UTC) |
|---|---|---|---|
| `ascend_quant_toolkit-0.4.2-py3-none-any.whl` | 41003 | `57964380f85275e5e7d32d14bc6df94c020d6be70bfe47da0dcc430d946d2470` | 2026-10-13T07:35:54.490867Z |
| `ascend_quant_toolkit-0.4.2.tar.gz` | 78146 | `5ff91e5343a8a3faaf687cac690ab7b90d4afa96a209514a3224765899548781` | 2026-10-13T07:35:55.653865Z |

## Publication controls, and the rejected first attempt

- The `pypi` environment carries a branch policy with custom deployment targets
  (`protected_branches=false`, `custom_branch_policies=true`), so a deployment is
  admitted only for a ref that matches an entry of that allow-list. The
  `toolkit-v*` tag namespace was not on it when the Release was first published.
- Attempt 1 (run
  [38358286519](https://github.com/jxd1111/Ascend-LLM-quant/actions/runs/38358286519), created 2026-10-13T06:43:05Z) passed the
  whole `Build and verify distributions` job and was then rejected before the
  upload job ran a single step:
  Tag "toolkit-v0.4.2" is not allowed to deploy to pypi due to environment protection rules. The deployment was rejected or didn't satisfy other protection rules.
- Nothing was uploaded by the rejected attempt, so the version was never burned:
  `https://pypi.org/pypi/ascend-quant-toolkit/json` answered `404` while PyPI held no
  `ascend-quant-toolkit` distribution at all. The fix was to extend the environment's
  deployment allow-list with the `toolkit-v*` tag pattern; the tag, the Release
  and the verified artifact were left untouched, and the rejected job was re-run
  as attempt 2.

## Cleanup and lifecycle

- The Toolkit distribution stays outside the serving process: the wheel carries
  `ascend_quant_toolkit` and the `ascend-quant-toolkit` console script only, and
  declares no runtime dependency.
- PyPI artifacts are immutable. A defect is corrected with a new version, or the
  release is yanked, according to project policy; a version is never re-used.

## Limitations

- The PyPI artifacts are not byte-reproducible across builds; this record lists
  the bytes the index actually serves rather than a locally rebuilt hash.
- This record was written after the release, so it is not part of the published
  0.4.2 sdist; the published revision remains `50ba3f8`.
- Only the offline Toolkit is covered here. The runtime counterpart
  (`vllm-ascend-quant-ext` 0.5.0) is recorded separately.

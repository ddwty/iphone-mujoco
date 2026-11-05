# Maintainer release workflow

GitHub source, downloadable artifacts and PyPI distribution are separate steps.
This project has a GitHub release-artifact workflow, not an automatic PyPI job.
Do not imply a release exists until its remote state and assets are verified.

## Before the first public push

1. Review the LICENSE and third-party notices. Keep optional vendor meshes out
   of Git; retain their original licenses in the model cache.
2. Run `python tools/check_repo_hygiene.py --mode source`. The public selection
   excludes local profiles, recordings, validation reports, environments,
   caches, generated scenes and distributions. It does not delete them.
3. Build a clean source archive with `python tools/export_source.py` and review
   its per-file SHA256 manifest. Inspect the actual staged index again with
   `python tools/check_repo_hygiene.py --mode staged` before committing.
4. Confirm GitHub authentication (`gh api user --jq .login`) belongs to the
   intended maintainer. Never paste a token into a source file or issue.
5. Create the repository without a generated README/license if pushing this
   existing main branch; avoid unrelated initial histories and force pushes.

For the current maintainer account, after local review:

```bash
gh repo create ddwty/iphone-mujoco --public --source . --remote origin --push
```

Use this once, only if the repository does not already exist. For an existing
remote, inspect its history and push a normal branch/PR instead of replacing it.

## Verify a candidate

```bash
python -m pip install -e '.[dev]'
python -m build
python tools/check_repo_hygiene.py --mode tracked
python tools/export_source.py
```

Run the focused checks listed in CONTRIBUTING.md and one graphical synthetic
smoke on a graphics-capable system. Inspect generated data with
`iphone-mujoco inspect-episode PATH --json`. Do not repeatedly run unrelated
tests or substitute a successful build for physical-device evidence.

The CI matrix installs the built wheel rather than importing the checkout.
One Linux job uses software OpenGL and validates a synthetic RGB-D observation.
After pushing, check the actual Actions result; workflow configuration alone
does not establish a passing Linux/macOS run.

## Publish GitHub artifacts

1. Update version in pyproject.toml, `iphone_mujoco.__version__`, capability
   metadata, CITATION.cff and CHANGELOG.md together.
2. Review the commit, then create/push the matching `vX.Y.Z` tag.
3. Prepare a GitHub Release with the relevant behavior, compatibility and test
   evidence. Review the draft before publishing.
4. Publishing the Release triggers the artifact workflow. It verifies that the
   tag equals the package version and attaches wheel, sdist, clean source ZIP,
   source manifest and checksums to that existing release.
5. Verify all remote assets/checksums and the selected commit. Large/private
   recordings are never release attachments by default.

```bash
git tag v0.3.0
git push origin v0.3.0
gh release create v0.3.0 --draft --generate-notes
```

Keep `v0.3.0` aligned with the actual version; the commands are not an instruction
to release an unreviewed working tree. Changing a published tag or rewriting
main history is not part of the normal release workflow.

## Repository settings

After the initial successful CI run, configure a main-branch ruleset requiring
the relevant checks and review, disallow force pushes, and enable private
vulnerability reporting. Keep Actions permissions minimal and review dependency
update PRs. A small project can use a single main branch and feature PRs; it
does not need a separate permanent development branch.

## Optional later PyPI publication

Confirm ownership/availability of the distribution name and configure PyPI
Trusted Publishing separately. Test the exact wheel/sdist first; use OIDC with
job-scoped permissions rather than committing API tokens. Do not rename the
underlying `asmagic` dependency or upload it as part of this project.

# Publishing to PyPI

This is the runbook for releasing a new version of anon-proxy. It assumes
maintainer access to the project's PyPI account and a clean checkout of `main`.

## Prerequisites (one-time)

1. **PyPI account** — create at https://pypi.org/account/register/.
2. **Test-PyPI account** — create at https://test.pypi.org/account/register/.
   Use a different password from prod PyPI.
3. **API tokens** — generate scoped-to-project tokens on both, store in your
   keychain. Never commit them.
4. **`twine` installed** — `uv tool install twine`. Or `pipx install twine`.

`~/.pypirc` (chmod 600):

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEI...   # your prod token

[testpypi]
username = __token__
password = pypi-AgEN...   # your test token
```

## Release procedure

1. **Bump the version** in `pyproject.toml` (`version = "X.Y.Z"`). Follow
   [SemVer](https://semver.org/): patch for bugfixes, minor for new features,
   major for breaking changes. Until 1.0, breaking changes can land in minors.
2. **Update CHANGELOG.md** if one exists — note user-visible changes.
3. **Commit and tag:**
   ```bash
   git commit -am "release: vX.Y.Z"
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin main vX.Y.Z
   ```
4. **Build the artifacts:**
   ```bash
   rm -rf dist/
   uv build
   ```
   Confirm `dist/` contains both the `.whl` and the `.tar.gz`.
5. **Smoke test the wheel locally:**
   ```bash
   python3 -m venv /tmp/release-smoke
   /tmp/release-smoke/bin/pip install dist/anon_proxy-X.Y.Z-py3-none-any.whl
   /tmp/release-smoke/bin/anon-proxy --help
   rm -rf /tmp/release-smoke
   ```
6. **Upload to test-pypi first:**
   ```bash
   twine upload --repository testpypi dist/*
   ```
   Visit https://test.pypi.org/project/anon-proxy/ and confirm the listing
   renders correctly (description, classifiers, links).
7. **Install from test-pypi to verify:**
   ```bash
   python3 -m venv /tmp/testpypi-smoke
   /tmp/testpypi-smoke/bin/pip install \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     anon-proxy
   /tmp/testpypi-smoke/bin/anon-proxy --help
   rm -rf /tmp/testpypi-smoke
   ```
8. **Upload to prod PyPI:**
   ```bash
   twine upload dist/*
   ```
9. **Post-release:**
   - Open a GitHub release at the tag with the changelog excerpt.
   - Bump version in `pyproject.toml` to next dev tag if you use one.
   - Tweet / post the announcement.

## If you publish a broken release

PyPI does not allow re-uploading the same version. You must:

1. Yank the broken release on the PyPI web UI (it stays installed for users
   who pinned it but disappears from search).
2. Bump to the next patch version and publish a fixed release.

## Common issues

- **`InvalidDistribution: Metadata is missing required fields`** — `pyproject.toml`
  is missing `name`, `version`, or `description`. Check Task 1 of the
  publishing-and-promotion plan.
- **`The user '__token__' isn't allowed to upload to project 'anon-proxy'`** —
  your token is scoped to a different project, or the project name was renamed.
  Generate a new token scoped to `anon-proxy`.
- **README renders as plain text on PyPI** — `readme = "README.md"` is set but
  PyPI couldn't determine the content type. Add to `[project]`:
  `readme = { file = "README.md", content-type = "text/markdown" }`.

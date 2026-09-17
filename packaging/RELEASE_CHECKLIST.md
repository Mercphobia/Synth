# Release Checklist

## Pre-Release
- [ ] Update version in `pyproject.toml`
- [ ] Update version in `src/synth/__init__.py`
- [ ] Update version in `packaging/npm-wrapper/package.json`
- [ ] Update version in `packaging/homebrew/synth.rb` (url and sha256)
- [ ] Run full test suite: `pytest tests/ --cov=src/synth`
- [ ] Verify coverage >= 85%
- [ ] Check all linters pass: `ruff check src/ tests/`
- [ ] Verify type checking: `mypy src/synth`
- [ ] Update `CHANGELOG.md` with release notes

## Build Artifacts
- [ ] Build wheel: `python -m build`
- [ ] Verify wheel contents: `tar -tzf dist/synth-*.whl`
- [ ] Test wheel installation: `pip install dist/synth-*.whl`
- [ ] Build binary: `./build_binary.sh`
- [ ] Test binary: `./dist/synth --version`
- [ ] Build Docker image: `docker build -t synth:latest .`
- [ ] Test Docker: `docker run synth:latest --version`

## Documentation
- [ ] Update README.md with new features
- [ ] Update CLI help text if commands changed
- [ ] Verify all examples still work
- [ ] Check API documentation is current

## Git Operations
- [ ] Commit all changes: `git add -A && git commit -m "Release vX.Y.Z"`
- [ ] Tag release: `git tag -a vX.Y.Z -m "Release X.Y.Z"`
- [ ] Push commits: `git push origin main`
- [ ] Push tag: `git push origin vX.Y.Z`

## GitHub Release
- [ ] Create release on GitHub
- [ ] Upload wheel: `dist/synth-*.whl`
- [ ] Upload binary: `dist/synth`
- [ ] Add release notes from CHANGELOG
- [ ] Verify all assets downloadable

## Post-Release
- [ ] Test npm package: `npm publish` (if applicable)
- [ ] Test Homebrew formula: `brew install synth` (if applicable)
- [ ] Verify Docker Hub image pushed (if automated)
- [ ] Announce release (if applicable)
- [ ] Update development version to next snapshot

## Verification Commands
```bash
# Test all installation methods
pip install synth-cli
synth --version

curl -fsSL https://raw.githubusercontent.com/Mercphobia/Synth/main/install.sh | sh
~/.synth-dist/bin/synth --version

npm install -g synth-cli
synth --version

brew install synth
synth --version

docker run synth:latest --version
```

# CONTRIBUTING

## Publishing a version

```
# update version in pyproject.toml
uv sync
git add pyproject.toml uv.lock
git commit -m 'bump version'
uv build --clear
git tag vX.Y.Z
git push --tags
uv publish --token "$TOKEN"
```

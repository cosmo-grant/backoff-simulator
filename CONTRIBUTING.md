# CONTRIBUTING

## Publishing a version

```
# update version in pyproject.toml and commit
uv build --clear
git tag vX.Y.Z
git push --tags
uv publish --token "$TOKEN"
```

# CONTRIBUTING

Release a version:

```
# update version in pyproject.toml
uv build --clear
git tag vX.Y.Z
git push --tags
uv publish --token "$TOKEN"
```

## Notes

- which python version you get on molab seems to vary, and not be configurable

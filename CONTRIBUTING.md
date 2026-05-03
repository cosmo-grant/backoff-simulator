# CONTRIBUTING

## Publishing a version

```
# update version in pyproject.toml and commit
uv build --clear
git tag vX.Y.Z
git push --tags
uv publish --token "$TOKEN"
```

## Modifying the notebook

Check locally: `uv run marimo edit notebook.py`.
Re-upload to molab.
Update notebook links in repo description and README.

## Adding more arguments

Add in `cli.py`, including help and validation.
Add in `notebook.py`.
Check everything matches (help, defaults, validation).

## Notes

Which python version you get on molab seems to vary, and not be configurable.

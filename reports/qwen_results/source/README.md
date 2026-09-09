# Local source files

The plotting script expects these files in this directory:

- `fixed-raw.json`
- `fixed-minhashlsh.json`
- `fixed-lshbloom.json`
- `efficiency-raw.json`
- `efficiency-minhashlsh.json`
- `fixed-sciq.csv`

The JSON files are the corresponding `result.json` and `curve-results.json`
objects under the project's S3 `pilot-5000` experiment prefix. They are
ignored by Git because they contain run and environment metadata. The derived
CSV tables, figures, and conclusions are versioned in the parent directory.

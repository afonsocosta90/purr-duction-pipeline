# How To Add New Data – "Am I a Cat?" Pipeline

**Production workflow** for extending the dataset while keeping the pipeline fully reproducible and validated.

## Current Dataset Status (Phase 2)
- **Cat**: 2400 images
- **Not Cat**: 4990 images
- **Total**: 7390 images
- Last validated via `validate` stage (metadata.csv automatically generated)

## Step-by-Step: Add New Data

### 1. Add your new images
Place new images in the raw data folders:
- `data/raw/cat/` → new cat images (any `.jpg`, `.jpeg`, or `.png`)
- `data/raw/not_cat/` → new not_cat images

You can add as many images as you want — there is no manual renaming required.

### 2. Re-run the full data pipeline

```bash
poetry run dvc repro --force
poetry run dvc push
````

### 3. What happens automatically

- `ingest.py` stage scans and organizes all images (old + new)
- `validate.py` stage runs quality gates (classes, balance, file integrity, deduplication)
- `build_features.py` stage (Phase 3) runs **stratified split** + saves resize/normalization config
- `data/processed/` is updated with:
  - `train.csv`, `val.csv`, `test.csv` (locked splits)
  - `features_config.json` (resize target + normalization stats)
- Pipeline fails early if any gate breaks

### Verify the update

```bash
poetry run dvc status
wc -l data/processed/{train.csv,val.csv,test.csv}   # check split sizes
cat data/processed/features_config.json | head -n 15
````

# Important Production Rules

- Never edit `data/processed/` manually
- Always use `dvc repro --force` after adding data
- Commit only `dvc.yaml` and `dvc.lock` (data itself is versioned by DVC)
- Run `poetry run dvc push` to share the new data with the team


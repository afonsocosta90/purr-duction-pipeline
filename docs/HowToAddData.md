# How To Add New Data – "Am I a Cat?" Pipeline

**Production workflow** for extending the dataset while keeping the pipeline fully reproducible and validated.

## Current Dataset Status
- **Cat**: 2,400 images
- **Not Cat**: 6,990 images
- **Total**: 9,390 images (imbalanced: 26 % / 74 %)
- Last validated via `validate` stage (metadata.csv automatically generated)

---

## How Image Classification Works (Critical)

The pipeline uses **filename capitalisation** as the sole signal to distinguish cat from non-cat images.
This convention comes from the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) and is enforced in `src/catops/data/ingest.py`.

| Filename first letter | Assigned class |
|-----------------------|----------------|
| **Uppercase** (`A`–`Z`) | `cat` |
| **Lowercase** (`a`–`z`) | `not_cat` |

**Examples**

```
Abyssinian_001.jpg   →  cat        ← Capital A
Bengal_003.jpg       →  cat        ← Capital B
golden_retriever_12.jpg  →  not_cat  ← lowercase g
labrador_007.jpg     →  not_cat    ← lowercase l
```

This rule is applied automatically by `ingest.py` — **you never touch `data/processed/` directly**.

---

## Step-by-Step: Add New Images

### 1. Name your files correctly

- **Cat image** → start the filename with an **Uppercase letter**
  - e.g. `Siamese_100.jpg`, `MaineCoon_042.png`
- **Not-cat image** → start the filename with a **lowercase letter**
  - e.g. `hamster_001.jpg`, `sunset_photo.png`

Accepted extensions: `.jpg`, `.jpeg`, `.png`

### 2. Place them in `data/raw/images/`

```
data/
└── raw/
    └── images/          ← drop new files here (flat directory, no subfolders)
        ├── Siamese_100.jpg      # will become → cat
        ├── MaineCoon_042.png    # will become → cat
        ├── hamster_001.jpg      # will become → not_cat
        └── ...
```

> If you are working from a fresh clone and the tar archive (`data/raw/images.tar.gz`) does not yet exist,
> run `poetry run dvc pull` first to restore the versioned data.

### 3. Re-run the full pipeline

```bash
poetry run dvc repro --force
poetry run dvc push
```

### 4. What happens automatically

| Stage | What it does |
|-------|-------------|
| `ingest.py` | Scans `data/raw/images/`, reads the first letter of every filename, copies cat images to `data/processed/cat/` and everything else to `data/processed/not_cat/` |
| `validate.py` | Quality gates: both classes present, each class ≥ 30 % of total, no file < 1 KB, no duplicate paths |
| `build_features.py` | Stratified 70/15/15 split → `train.csv`, `val.csv`, `test.csv`; saves `features_config.json` (resize target + normalisation stats) |
| DVC | Writes `dvc.lock` (content hashes) for full reproducibility |

The pipeline raises an error and stops early if any quality gate fails.

### 5. Verify the update

```bash
poetry run dvc status                               # should be "Data and pipelines are up to date"
wc -l data/processed/{train.csv,val.csv,test.csv}  # check new split sizes
cat data/processed/features_config.json | head -n 15
```

### 6. Commit

```bash
git add dvc.lock
git commit -m "data: add <N> new images (cat: X, not_cat: Y)"
git push
```

---

## Important Production Rules

| Rule | Reason |
|------|--------|
| Never edit `data/processed/` manually | That directory is fully owned by the pipeline |
| Always `dvc repro --force` after adding data | Forces all downstream stages to recompute |
| Commit only `dvc.yaml` and `dvc.lock` | The data itself is version-controlled by DVC, not Git |
| Run `poetry run dvc push` before pushing your Git branch | Teammates need the new data artifacts in the remote |
| Respect the capitalisation rule | A lowercase cat image will silently be labelled `not_cat` |

---

## Troubleshooting

**My new cat image ended up in `not_cat`.**  
Check that the filename starts with an uppercase letter. Rename it and re-run the pipeline.

**Validation gate failed: class imbalance.**  
The pipeline requires each class to be at least 30 % of the total. If you added many images of only one class, add complementary images for the other class before re-running.

**`FileNotFoundError: Dataset not found at data/raw/images.tar.gz`**  
Run `poetry run dvc pull` to restore the versioned archive from the DVC remote before adding new files.

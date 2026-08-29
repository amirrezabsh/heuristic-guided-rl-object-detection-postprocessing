# Datasets

Datasets are not committed to Git. The local experiments use YOLO-format YAML
files under `datasets/`, and those files often contain absolute paths.

## Global Wheat

Global Wheat is the primary benchmark because it is outside COCO/VOC and uses a
domain-specific object class.

Expected local files:

```text
datasets/globalwheat_subsets/pilot.yaml
datasets/globalwheat_subsets/full.yaml
```

The final pilot benchmark used:

```text
train-limit = 200
eval-limit  = 200
seeds       = 0, 1, 2
```

The repository includes scripts for training and evaluating the policy, but not
the dataset itself.

## Optional Additional Benchmarks

Useful non-COCO/VOC datasets to add:

- SKU-110K: retail product detection
- Hard Hat Workers: helmet/head/person safety detection

After downloading and converting a dataset to YOLO format, create:

```text
datasets/<dataset>_subsets/pilot.yaml
datasets/<dataset>_subsets/full.yaml
```

Then run `scripts/train_continuous_postprocess_rl.py` with the corresponding
detector weights and YAML file.

## Final Benchmark Datasets

SKU-110K and Hard Hat Workers are the datasets used by the guarded final
benchmark workflow. Each preparation script produces deterministic YOLO split
lists, `pilot.yaml`, `full.yaml`, and `metadata.json`. The source dataset is
not committed; keep it outside version control and pass its extracted root on
the command line.

For SKU-110K, the extracted `SKU110K_fixed` root must contain `images/` and
`annotations/`:

```bash
.venv/bin/python scripts/prepare_sku110k.py \
  --root /path/to/SKU110K_fixed \
  --out datasets/sku110k_subsets
```

For Hard Hat Workers, the extracted root must contain `images/` and Pascal VOC
XML files in `annotations/`:

```bash
.venv/bin/python scripts/prepare_hardhat.py \
  --root /path/to/hardhat_workers \
  --out datasets/hardhat_subsets
```

The final runner accepts dataset locations through environment variables. Set
them explicitly rather than relying on machine-specific defaults:

```bash
SKU_DATA="$PWD/datasets/sku110k_subsets/full.yaml" \
HARDHAT_DATA="$PWD/datasets/hardhat_subsets/full.yaml" \
bash scripts/run_final_new_dataset_benchmarks.sh status
```

Continue with [FINAL_TESTS.md](../FINAL_TESTS.md) for the fixed protocol and
the commands that train, validate, and perform the one-time final evaluation.

## Notes On COCO And VOC

COCO and VOC are useful for sanity checks and detector-family comparisons, but
they should not be presented as the primary novelty benchmark when using
detectors pretrained on COCO-style data.

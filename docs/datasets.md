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

## Notes On COCO And VOC

COCO and VOC are useful for sanity checks and detector-family comparisons, but
they should not be presented as the primary novelty benchmark when using
detectors pretrained on COCO-style data.

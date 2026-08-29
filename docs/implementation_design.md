# Implementation Design

This repository contains two research tracks from the thesis work:

1. an earlier RL-NAS / layer-wise fine-tuning scaffold
2. the final adaptive post-processing method

The final thesis contribution is the second track.

## Final Track: Adaptive Post-Processing

The final method keeps an object detector frozen and adapts post-processing per
image. The implementation is centered on:

```text
scripts/train_postprocess_policy.py
scripts/train_continuous_postprocess_rl.py
scripts/train_torchvision_continuous_postprocess_rl.py
scripts/summarize_postprocess_rl.py
```

### Core Flow

1. Cache detector candidate predictions at permissive thresholds.
2. Build per-image state vectors from prediction statistics.
3. Select post-processing actions.
4. Re-run thresholding/NMS locally.
5. Compute mAP-style reward against ground truth.
6. Update a small policy with REINFORCE.

### Key Components

`ImageRecord`

Stores one image's ground truth and cached detector predictions.

`Action`

Stores post-processing controls:

```text
conf
iou
max_det
agnostic
min_area
top_k
```

The final compact method uses only `conf`, `iou`, and `max_det`.

`ContinuousPostprocessPolicy`

A small MLP that outputs a Gaussian policy over continuous action values.

`heuristic_actions`

Hand-written adaptive baselines using candidate density, confidence, and box
overlap statistics.

`residual_combined` mode

Uses the combined heuristic action as an action prior and trains RL to produce a
bounded residual correction.

## Earlier Track: RL-NAS / Fine-Tuning Scaffold

The `src/rlnas_od` and `src/rl_finetune` packages contain the earlier
experiments around architecture search and layer-wise learning-rate control.
They are retained for transparency and because some tests cover them, but they
are not the final thesis claim.

Important files:

```text
src/rlnas_od/
src/rl_finetune/
scripts/train_layerwise_rl.py
scripts/train_multiepisode_policy.py
```

## Final Benchmark Orchestration

The final SKU-110K and Hard Hat Workers workflow separates dataset preparation,
training, and held-out evaluation so that test data cannot be used while
selecting a detector or policy:

- `prepare_sku110k.py` and `prepare_hardhat.py` create deterministic YOLO
  datasets and split manifests.
- `train_yolo_detector.py` fine-tunes a detector in resumable chunks, including
  safeguards for Apple Silicon memory pressure.
- `run_final_new_dataset_benchmarks.sh train` trains detectors and policies on
  train/validation data only, and `status` checks readiness without evaluating
  the test split.
- `evaluate_frozen_postprocess_policies.py` records hashes for the detector,
  policies, dataset YAML, and split list, then refuses a second completed final
  evaluation.
- `summarize_final_new_dataset_tests.py` combines the per-dataset outputs into
  a thesis-ready comparison table and summary.

See [FINAL_TESTS.md](../FINAL_TESTS.md) for the operational protocol and
[datasets.md](datasets.md) for source-layout and preparation requirements.

## Design Lessons

The final experiments showed:

- direct RL improves fixed/random baselines but is not strongest alone
- heuristic adaptive post-processing is a strong baseline
- residual RL on top of the heuristic gives the best compact result
- adding more action dimensions can destabilize learning
- curriculum training recovers extended-action performance but does not beat the
  compact action space

These observations motivate the final design choice: compact heuristic-guided
residual RL.

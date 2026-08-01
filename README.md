# Heuristic-Guided RL for Adaptive Object Detection Post-Processing

This repository contains the implementation and experiments for an MSc thesis
project on adapting pretrained object detectors with lightweight reinforcement
learning.

The final method is **heuristic-guided residual contextual-bandit RL** for
object detection post-processing. A frozen detector first produces candidate
boxes. A small policy then adapts post-processing parameters per image:

- confidence threshold
- NMS IoU threshold
- maximum detections

The strongest variant starts from a hand-written adaptive heuristic and trains a
REINFORCE policy to learn residual corrections around that heuristic.

## Research Question

Can reinforcement learning improve the inference behavior of pretrained object
detectors without retraining detector weights?

The study compares:

1. fixed standard post-processing
2. random adaptive post-processing
3. direct RL post-processing
4. hand-written heuristic adaptive post-processing
5. heuristic-guided residual RL
6. compact vs expanded action spaces

## Main Result

On the Global Wheat benchmark subset, the best compact residual RL method
achieved the highest mAP50-95:

| Method | mAP50-95 |
|---|---:|
| Standard fixed post-processing | 0.372817 |
| Direct continuous RL | 0.386348 |
| Heuristic combined rule | 0.394221 |
| Heuristic-guided residual RL | 0.397244 |
| Naive extended action RL | 0.376177 |
| Curriculum extended action RL | 0.393841 |

The results show that direct RL from scratch is weaker than a strong heuristic,
but RL is useful as a residual optimizer on top of the heuristic prior.

## Repository Layout

```text
src/
  rlnas_od/          Earlier RL architecture-search scaffold and utilities
  rl_finetune/       Layer-wise RL fine-tuning utilities

scripts/
  train_continuous_postprocess_rl.py
  train_postprocess_policy.py
  train_torchvision_continuous_postprocess_rl.py
  summarize_postprocess_rl.py
  add_heuristic_postprocess_metrics.py

docs/
  method.md
  results.md
  datasets.md
  implementation_design.md

tests/
  Unit tests for the earlier search/fine-tuning utilities
```

Generated artifacts such as `runs/`, downloaded datasets, pretrained weights,
and virtual environments are intentionally excluded from Git.

## Setup

Python 3.9+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Apple Silicon experiments, use `device=mps` where supported. TorchVision
detection models are more reliable on CPU in this project.

## Dataset Preparation

The main benchmark used in the final result is Global Wheat. Dataset files are
not committed because they are large and contain local absolute paths. See
[docs/datasets.md](docs/datasets.md) for dataset setup notes.

Expected local YAML examples:

```text
datasets/globalwheat_subsets/pilot.yaml
datasets/globalwheat_subsets/full.yaml
```

## Reproducing The Main Global Wheat Experiment

Train/evaluate compact residual RL on the Global Wheat subset:

```bash
for seed in 0 1 2; do
  .venv/bin/python scripts/train_continuous_postprocess_rl.py \
    --model runs/globalwheat_yolo_supervised/yolov8n_pilot_e20/weights/best.pt \
    --data datasets/globalwheat_subsets/pilot.yaml \
    --train-split train \
    --eval-split test \
    --train-limit 200 \
    --eval-limit 200 \
    --imgsz 416 \
    --batch 4 \
    --device mps \
    --policy-epochs 40 \
    --rl-batch-size 16 \
    --seed "$seed" \
    --project runs/globalwheat_residual_state_rl \
    --name "seed${seed}_e40_rs020" \
    --action-mode residual_combined \
    --residual-scale 0.20 \
    --action-space base
done
```

Summarize:

```bash
.venv/bin/python scripts/summarize_postprocess_rl.py \
  --root runs/globalwheat_residual_state_rl \
  --min-eval-images 100
```

## Useful Ablations

Direct RL from scratch:

```bash
.venv/bin/python scripts/train_continuous_postprocess_rl.py \
  --action-mode direct \
  --action-space base
```

Naive expanded action space:

```bash
.venv/bin/python scripts/train_continuous_postprocess_rl.py \
  --action-mode residual_combined \
  --action-space extended \
  --min-area-range 0.0,0.002 \
  --top-k-range 200,1000
```

Curriculum expanded action space:

```bash
.venv/bin/python scripts/train_continuous_postprocess_rl.py \
  --action-mode residual_combined \
  --action-space extended \
  --freeze-extended-epochs 30 \
  --min-area-range 0.0,0.001 \
  --top-k-range 400,1000
```

## Tests

The original NAS/fine-tuning utilities have unit tests:

```bash
python -m pytest tests
```

The post-processing experiments are validated through smoke runs and summary
files because they depend on local datasets and detector weights.

## Thesis Framing

The final contribution should be stated conservatively:

> A heuristic-guided residual contextual-bandit RL policy can improve adaptive
> post-processing for frozen pretrained object detectors, but pure RL from
> scratch is not the strongest method under limited compute.

This repository is intended as reproducible research code, not a production
object detection package.

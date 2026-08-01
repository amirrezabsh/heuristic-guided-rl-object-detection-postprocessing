# Results

All values below are mAP50-95.

## Global Wheat Main Ablation

| Method | Mean mAP50-95 | Notes |
|---|---:|---|
| Standard fixed post-processing | 0.372817 | YOLO default-style fixed thresholds |
| Random continuous | 0.375496 | Per-image random continuous actions |
| Direct continuous RL | 0.386348 | RL from scratch |
| Heuristic density | 0.364515 | Hand-written density rule |
| Heuristic confidence | 0.389588 | Hand-written confidence rule |
| Heuristic combined | 0.394221 | Density + confidence + overlap |
| Residual heuristic RL | 0.397244 | Best compact-action result |

## Action-Space Ablation

| Method | Mean mAP50-95 | Interpretation |
|---|---:|---|
| Compact residual RL | 0.397244 | Best result |
| Naive extended residual RL | 0.376177 | Larger action space destabilized learning |
| Curriculum extended residual RL | 0.393841 | Recovered performance but did not beat compact |

## COCO Subset Cross-Check

| Method | Mean mAP50-95 |
|---|---:|
| Standard | 0.336613 |
| Random continuous | 0.353227 |
| Direct continuous RL | 0.358682 |
| Heuristic confidence | 0.387960 |

COCO is reported as a secondary cross-check because YOLOv8n is pretrained on
COCO-like data. It is not the primary thesis benchmark.

## TorchVision Detector Cross-Check

| Detector | Standard | Direct RL | Best Heuristic |
|---|---:|---:|---:|
| Faster R-CNN MobileNetV3-FPN | 0.404445 | 0.408518 | 0.415440 |
| SSDLite320 MobileNetV3 | 0.271370 | 0.292864 | 0.304818 |

These results show that the post-processing framework is detector-agnostic, but
the strongest final claim is based on Global Wheat compact residual RL.

## Main Takeaway

Direct RL improves over fixed/random baselines, but a hand-written heuristic is
stronger. The final contribution is therefore not pure RL from scratch. The best
result comes from using the heuristic as a prior and training RL to learn
residual corrections.

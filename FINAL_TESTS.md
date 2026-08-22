# Final tests on SKU-110K and Hard Hat Workers

The workflow has two deliberately separate stages. The first stage may inspect
`train` and `val`; the second stage opens each complete `test` split once using
already-frozen detector weights and policies.

## 1. Confirm the fixed protocol

The runner defaults are:

- YOLOv8n, 20 detector fine-tuning epochs
- image size 416 and MPS device
- batch size 1 for dense SKU-110K and batch size 4 for Hard Hat Workers
- 1,000 policy-training images and 500 validation images
- compact combined-heuristic residual action space
- residual scale 0.25
- 40 policy epochs and seeds 0, 1, and 2
- complete test splits (`eval-limit=0`) for the final evaluation

Do not change these defaults between the training and final-test commands.

## 2. Train and validate without opening test

For clean final results, restart macOS first. Then open only Terminal and close
Telegram, Chrome, VS Code, Docker/virtual machines, ChatGPT/Codex, App Store,
and other Electron applications. From the repository root run:

```bash
caffeinate -dimsu bash scripts/run_final_new_dataset_benchmarks.sh train
```

The runner performs a memory preflight and refuses to start with less than 5 GB
available or more than 4 GB of swap already occupied. Do not bypass this check
for the official run.

This command is resumable. Existing complete detector weights and policies are
skipped. An interrupted detector resumes from `last.pt`; `best.pt` alone is not
treated as completion. The SKU-110K process periodically releases the MPS cache
and uses a conservative memory limit. Detector training runs one complete epoch
per process, then relaunches from `last.pt`, preventing long-lived MPS memory
growth. It produces validation summaries at:

```text
runs/final_sku110k_postprocess_rl_b1_clean/summary.csv
runs/final_hardhat_postprocess_rl_b4_clean/summary.csv
```

The earlier mixed-batch SKU run remains preserved at
`runs/final_sku110k_yolo_supervised/yolov8n_e20/` as a failed pilot. The
official workflow never reads it. The clean SKU run starts at epoch 0 under
`yolov8n_e20_b1_chunk1_clean/`.

Check readiness without evaluating test:

```bash
bash scripts/run_final_new_dataset_benchmarks.sh status
```

Before continuing, both detectors and all six policies must show `[ready]`, and
both final-test markers must show `[not-run]`.

## 3. Freeze the protocol and run the final tests once

Do not tune anything after this command. Run:

```bash
CONFIRM_FINAL_TEST=YES_I_HAVE_FROZEN_THE_PROTOCOL \
  bash scripts/run_final_new_dataset_benchmarks.sh final-test
```

The evaluator preflights both datasets, both detector checkpoints, all policy
metadata, and policy seeds 0/1/2 before opening either test split. Detector
candidate predictions are cached once per image and reused for Standard,
Combined Heuristic, and all three frozen Residual-RL policies.

If the process is interrupted, rerun the exact same command. Completed batches
resume from the prediction cache, and a dataset with a completion marker is
skipped. Changed protocols are rejected.

## 4. Final outputs

Per-dataset results:

```text
runs/final_sku110k_test/comparison.csv
runs/final_sku110k_test/summary.json
runs/final_hardhat_test/comparison.csv
runs/final_hardhat_test/summary.json
```

Combined thesis-ready results:

```text
runs/final_new_dataset_tests/comparison.csv
runs/final_new_dataset_tests/summary.json
```

The completion markers include SHA-256 hashes of the detector, policies,
dataset YAML, and split list. Once `FINAL_TEST_COMPLETE.json` exists, the
evaluator refuses to run that final test again.

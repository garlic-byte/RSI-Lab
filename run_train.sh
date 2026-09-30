#!/bin/bash
cd /Users/garlic/Desktop/codex/model
/Users/garlic/Desktop/venv/torch/bin/python run_train.py \
  --device mps \
  --output-dir outputs/experiment_01 \
  --save-progress \
  --log-every 200 \
  --save-every 1000
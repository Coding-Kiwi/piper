#!/usr/bin/env bash
set -eo pipefail

cd /app

./build_monotonic_align.sh

echo "start training"

python3 -m piper_train \
    --dataset-dir /app/dataset-prepared/ \
    --accelerator 'gpu' \
    --devices 1 \
    --batch-size 16 \
    --validation-split 0.0 \
    --num-test-examples 0 \
    --max_epochs 10000 \
    --resume_from_checkpoint /app/thorsten.ckpt \
    --checkpoint-epochs 1 \
    --precision 32 \
    --quality high

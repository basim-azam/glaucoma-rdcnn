#!/bin/bash
# Drop into a 4-hour interactive A100 session. Defaults to gpu-a100-short for fast scheduling.
sinteractive \
    --partition=gpu-a100-short \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=04:00:00

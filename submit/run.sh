#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python submit/run.py \
    --logdir-start=./submit/checkpoints/start \
    --logdir-run=./submit/checkpoints/run \
    --logdir-side=./submit/checkpoints/side \
    --outdir=./submit/logs \
    --n-cpu=1 --visualize

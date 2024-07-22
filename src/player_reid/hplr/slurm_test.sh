#!/bin/bash

sbatch \
    --gpus 8 \
    -o $ENDPOINT/checkpoint/dir/%j.out \
    -J levi-test \
    --wrap="python _d.py"
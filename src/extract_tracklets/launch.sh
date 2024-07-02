#!/bin/bash

# This script launches pipeline.py 8 times with rank values from 0 to 7

for rank in {0..0}
do
    # nohup python3 pipeline.py --rank $rank &
    python3 pipeline.py --rank $rank &
done

# Wait for all background processes to complete
wait

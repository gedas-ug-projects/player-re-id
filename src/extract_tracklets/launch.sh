#!/bin/bash

for rank in {0..7}
do
    nohup python3 pipeline.py --rank $rank &
    # python3 pipeline.py --rank $rank &
done

# Wait for all background processes to complete
wait

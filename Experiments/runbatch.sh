#!/usr/bin/env bash
COUNTER=0
while [  $COUNTER -lt 10 ]; do
     python WindPredictionRNNBatchDB.py --best --nbatches 3
     echo Run number $COUNTER
     let COUNTER=COUNTER+1
done

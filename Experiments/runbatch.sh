#!/usr/bin/env bash
COUNTER=0
while [  $COUNTER -lt 30 ]; do
     python WindPredictionRNNBatchDB.py --best --nbatches 1
     echo Run number $COUNTER
     let COUNTER=COUNTER+1
done
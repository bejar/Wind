#!/usr/bin/env bash
COUNTER=0
while [  $COUNTER -lt $1 ]; do
     python WindPredictionRNNBatchDB.py --best --proxy --early --gpu
     echo Run number $COUNTER
     let COUNTER=COUNTER+1
done

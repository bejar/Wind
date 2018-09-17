#!/usr/bin/env bash
COUNTER=0
while [  $COUNTER -lt $1 ]; do
     python WindPredictionBatch.py --best --early  --exp mlpregs2s
     echo Run number $COUNTER
     let COUNTER=COUNTER+1
done

#!/bin/bash
# train.sh
weaver \
  --data-train "$DATA_PATH/*.root" \
  --data-config config/pfjet_tagger.yaml \
  --network-config models/topTagger_model.py \
  --model-prefix output/training \
  --num-workers 2 \
  --gpus 0 \
  --batch-size 32 \
  --start-lr 1e-5 \
  --num-epochs 20 \
  --optimizer ranger \
  --fetch-step 0.01 

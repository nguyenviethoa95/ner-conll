#!/usr/bin/env bash
python3 inference.py  --embedding-dim=50 \
                      --embedding-path=data/glove.6B.50d/glove.6B.50d.txt  \
                      --num-layers=1 \
                      --hidden-size=100     \
                      --output_dim=9  \
                      --gpu=cuda:1 \
                      --model-dir=model_files/ \
                      --model-name=model_20211217 \
                      --test-data-dir=data/ \
                      --test-data-file=test.conll
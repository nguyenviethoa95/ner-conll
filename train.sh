#!/usr/bin/env bash
python3 train.py --embedding-dim=50 \
                  --embedding-path=data/glove.6B.50d/glove.6B.50d.txt  \
                  --batch-size=1     \
                  --hidden-size=100     \
                  --output_dim=9  \
                  --learning-rate=2e-05 \
                  --epochs=20 \
                  --num-layers=1 \
                  --gpu=cuda:2 \
                  --model-dir=model_files/ \
                  --model-name=model_20211217 \
                  --training-data-dir=data/ \
                  --training-data-file=train.conll \
                  --validation-data-dir=data/ \
                  --validation-data-file=dev.conll
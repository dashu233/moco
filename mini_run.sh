#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_moco.py data/imagenet \
  -a resnet50 --use_pretrained_model '' --mini_train True \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --cos --aug-plus --lr 0.015 --batch-size 128 \
  --output output/my_mini_pretrained_model --prune_steps [300] --epoch 200 \
  | tee -a stdout.txt

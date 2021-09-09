#!/bin/sh

CUDA_VISIBLE_DEVICES=4,5,6,7 python main_moco.py data/imagenet \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --lr 0.0003 --batch-size 256 --mini_train True \
  --output output/mini_pretrained_exp0 --prune_steps [10,20,30,40,50,60,70,80,90] --epoch 100 \
  | tee -a stdout.txt

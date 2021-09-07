#!/bin/sh

CUDA_VISIBLE_DEVICES=4,5,6,7 python main_moco.py data/imagenet \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos --lr 0.003 --batch-size 256 --resume output/pretrained_exp3/checkpoint_0009.pth.tar \
  --output output/pretrained_exp4 --prune_steps [10,30,50,70] --epoch 100 --schedule 40 80 \
  | tee -a stdout.txt

#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_lincls.py data/imagenet \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained output/pretrained_exp3/checkpoint_0020.pth.tar \
  --output output/pretrained_exp3_lin \
  | tee -a stdout_lin.txt

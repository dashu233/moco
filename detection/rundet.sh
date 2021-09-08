#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_net.py --config-file configs/coco_R_50_C4_2x_moco.yaml \
	--num-gpus 4 MODEL.WEIGHTS ./output.pkl

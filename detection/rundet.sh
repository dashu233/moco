#!/bin/sh

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python train_net.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml \
	--num-gpus 4 MODEL.WEIGHTS output100.pkl OUTPUT_DIR output/02x0_prune_VOC_100extraepoch

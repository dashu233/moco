#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle as pkl
import sys
import torch

if __name__ == "__main__":
    start_key = "module.encoder_q."
    print('using {} as the backbone'.format(start_key))
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    obj = obj["state_dict"]

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith(start_key):
            continue
        old_k = k
        k = k.replace(start_key, "")
        if "layer" not in k:
            k = "stem." + k

        if ('conv' in k or 'shortcut' in k) and not 'bn' in k and 'weight' in k and 'weight_orig' not in k and 'weight_mask' not in k:
            k = k.replace('weight','weight_orig')

        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = v.numpy()

    res = {"model": newmodel, "__author__": "MOCO", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)

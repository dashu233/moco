from posixpath import join
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch
import os
import shutil
import pathspec


@torch.no_grad()
def para_count_conv_mask(model:nn.Module):
    para_orig = 0
    para_remain = 0
    for m in model.modules:
        if hasattr(m,'bias'):
                num = torch.numel(m.bias)
                para_remain += num
                para_orig += num

        if isinstance(m,nn.Conv2d):
            assert hasattr(m,'weight_mask'), 'make sure the model has been pruned!'
            para_remain += int(torch.sum(m.weight_mask))
            para_orig += int(torch.numel(m.weight_mask))
        else:
            if hasattr(m,'weight'):
                num = torch.numel(m.weight)
                para_remain += num
                para_orig += num
    return para_remain,para_orig

def backup_code(root_dir, obj_dir):
    """
    Backing up code with git
    :param root_dir: project root directory
    :param obj_dir: object directory which stores backup files
    """
    print('backup coding...')
    root_dir = os.path.realpath(root_dir)
    obj_dir = os.path.realpath(obj_dir)
    os.makedirs(obj_dir, exist_ok=True)
    ignore_path = os.path.join(root_dir, ".gitignore")
    assert os.path.exists(ignore_path), "There must be a .gitignore file!!!"
    with open(ignore_path, 'r') as fp:
        spec = pathspec.PathSpec.from_lines('gitwildmatch', fp)
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            ori_file = os.path.join(root, f)
            dst_file = obj_dir + ori_file[len(root_dir):]
            if spec.match_file(ori_file):
                continue
            folder, _ = os.path.split(dst_file)
            os.makedirs(folder, exist_ok=True)
            shutil.copy(ori_file, dst_file)
            
    print('finish backup')
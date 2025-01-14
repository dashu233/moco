#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import json
import moco.loader
import moco.builder
import sys
from torch.utils.tensorboard import SummaryWriter
from utils import backup_code,para_count_conv_mask
import augment
from torch.profiler import profile, record_function, ProfilerActivity


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--output',type = str, default='output',help = 'output_checkpoint_dir')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')


# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


# my parser
def str2bool(a:str):
    return a.lower() == 'true'

parser.add_argument('--prune_rate',type = float, default=0.2)
parser.add_argument('--prune_steps',type = str, default='[10]')
parser.add_argument('--prune_method',type = str, default='IMP')
parser.add_argument('--mask_encode',type=str,choices=['query','key','all'],default='query')
parser.add_argument('--unstructure',type=str2bool,default='True')
parser.add_argument('--mask_module',type=str,choices=['conv','bn'],default='conv')
parser.add_argument('--use_pretrained_model',type=str,default='moco_v2_200ep_pretrain.pth.tar')
parser.add_argument('--mini_train',type=str2bool,default='False')
parser.add_argument('--use_trans_match',type=str2bool,default='False')
parser.add_argument('--reg_loss_penalty',type=float,default=1e-4)
parser.add_argument('--checkpoint_interval',type=int,default=10)
parser.add_argument('--l1_penalty',type=float,default=0.001)


def deal_with_args(arg):
    arg.prune_steps = json.loads(arg.prune_steps)
    return arg
def get_prune_rate(arg,ep):
    return arg.prune_rate
    
def main():
    args = parser.parse_args()
    args = deal_with_args(args)
    print(args.prune_steps)
    print(args.schedule)
    
    if os.path.exists(args.output):
        print('an existed dir, enter e or exit to cancel the command, or anything else to continue')
        st = input()
        if st in ['e','exit']:
            exit()
    else:
        os.mkdir(args.output)
    
    if not os.path.exists(os.path.join(args.output,'log')):
        os.mkdir(os.path.join(args.output,'log'))
    if not os.path.exists(os.path.join(args.output,'backup')):
        os.mkdir(os.path.join(args.output,'backup'))
    if not os.path.exists('history'):
        os.mkdir('history')
    now_time = time.strftime("%a-%b-%d-%H-%M-%S-%Y", time.localtime()) 
    if not os.path.exists(os.path.join('history',now_time)):
        os.mkdir(os.path.join('history',now_time))
    
    backup_code('./',os.path.join(args.output,'backup'))
    backup_code('./',os.path.join('history',now_time))


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def is_main_process(args):
    if torch.distributed.is_initialized():
        return args.rank % torch.cuda.device_count()
    else:
        return True

def main_worker(gpu, ngpus_per_node, args):
    if is_main_process(args):
        writer = SummaryWriter(os.path.join(args.output,'log'))
    else:
        writer = None
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if not args.use_trans_match:
        model = moco.builder.MoCoUnstructruedPruned(
            models.__dict__[args.arch],args,
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    else:
        model = moco.builder.CAMMoCo(
            models.__dict__[args.arch],args,
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    print(model)
    if args.use_pretrained_model:
        if os.path.isfile(args.use_pretrained_model):
            print("=> loading checkpoint '{}'".format(args.use_pretrained_model))
            if args.gpu is None:
                checkpoint = torch.load(args.use_pretrained_model)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.use_pretrained_model, map_location=loc)
            cp = {}
            for ky in checkpoint['state_dict']:
                cp[ky[7:]] = checkpoint['state_dict'][ky]
            #print('before:',model.queue)
            #print('keys in cp',cp.keys())
            if not args.use_trans_match:
                st = model.state_dict()
                st.update(cp)
                model.load_state_dict(st)
                model.key_copy()
            else:
                model.load_state_dict_from_MoCo(cp)

            print("=> use pretrained model '{}' (epoch {})"
                  .format(args.use_pretrained_model, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.use_pretrained_model))

    model.add_prune_mask()
    # optionally resume from a checkpoint
    
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    # Data loading code
    if args.mini_train:
        traindir = os.path.join(args.data, 'mini_train')
    else:
        traindir = os.path.join(args.data, 'train')
    
    if not args.use_trans_match:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if args.aug_plus:
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            augmentation = [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        else:
            # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
            augmentation = [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]

        train_dataset = datasets.ImageFolder(
            traindir,
            moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    else:
        augmentation = augment.AugPlus()
        train_dataset = datasets.ImageFolder(
            traindir,
            augmentation
        )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    remain_rate = 1
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        if args.use_trans_match:
            train_trans(train_loader, model, criterion, optimizer, epoch, args,writer)
        else:
            train(train_loader, model, criterion, optimizer, epoch, args,writer)
        if epoch in args.prune_steps:
            remain_rate *= (1-args.prune_rate)
            print('remain_percent:',remain_rate)
            prune_rate = get_prune_rate(args,epoch)
            model.module.prune_step(prune_rate)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename= os.path.join(args.output,'last_model.pth.tar'.format(epoch)))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0) and (epoch%args.checkpoint_interval == args.checkpoint_interval-1):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename= os.path.join(args.output,'checkpoint_{:04d}.pth.tar'.format(epoch)))


def train(train_loader, model, criterion, optimizer, epoch, args,writer:SummaryWriter):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    #with record_function('train_loop'):
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        #with record_function('model_forward'):
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        #with record_function('model_backward'):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
            sys.stdout.flush()
    if writer is not None:
        writer.add_scalar('loss',losses.avg,epoch)
        writer.add_scalar('top1',top1.avg,epoch)
        writer.add_scalar('prune_rate',get_prune_rate(args,epoch),epoch)



def train_trans(train_loader, model, criterion, optimizer, epoch, args,writer:SummaryWriter):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    reg_losses = AverageMeter('RegLoss', ':.4e')
    cls_losses = AverageMeter('ClsLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, reg_losses, cls_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            images[2] = images[2].cuda(args.gpu, non_blocking=True)
            images[3] = images[3].cuda(args.gpu, non_blocking=True)
        # compute output
        output, target,reg_loss = model(im_q=images[0], im_k=images[1],trans_q=images[2],trans_k=images[3])
        cls_loss = criterion(output, target)
        loss = cls_loss + args.reg_loss_penalty * reg_loss

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        reg_losses.update(reg_loss.item(),images[0].size(0))
        cls_losses.update(cls_loss.item(),images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))
        
        # print("top1:{}".format(top1.avg))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        model.module.add_l1_loss(args.l1_penalty)
        # for name, param in model.module.named_parameters():
        #     if param.grad is None:
        #         print(name)

        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            #print('diff:')
            #print(output[:,0]-torch.max(output,dim=1).values)
            # for name,m in model.module.encoder_q.named_parameters():
            #     if m.grad is not None:
            #         print(name,torch.norm(m.grad))
            # for name,m in model.module.head.named_parameters():
            #     if m.grad is not None:
            #         print(name,torch.norm(m.grad))
            progress.display(i)
            sys.stdout.flush()
    if writer is not None:
        writer.add_scalar('loss',losses.avg,epoch)
        writer.add_scalar('cls_loss',cls_losses.avg,epoch)
        writer.add_scalar('reg_loss',reg_losses.avg,epoch)
        writer.add_scalar('top1',top1.avg,epoch)
        writer.add_scalar('prune_rate',get_prune_rate(args,epoch),epoch)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    print('learning rate:',lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    #with profile(activities=[
    #    ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #    with record_function('main_process'):
    main()
    #print(prof.key_averages().table(
    #    sort_by="self_cuda_time_total", row_limit=-1))
    

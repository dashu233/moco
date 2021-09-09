# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from numpy.lib.financial import ipmt
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

class Identity(nn.Module):
    def forward(self,x):
        return x

class MoCoUnstructruedPruned(MoCo):
    def __init__(self, base_encoder, args,dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        super().__init__(base_encoder, dim=dim, K=K, m=m, T=T, mlp=mlp)
        self.args = args
        self.parameters_to_prune = []
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder, mask value will goes to zero
        """
        for name,para in self.encoder_q.named_parameters():
            if '_orig' in name:
                self.encoder_k.state_dict()[name].data = self.encoder_k.state_dict()[name].data * self.m \
                    + torch.mul(para.data, self.encoder_q.state_dict()[name.replace('_orig','_mask')]) * (1. - self.m)
            else:
                self.encoder_k.state_dict()[name].data = self.encoder_k.state_dict()[name].data * self.m \
                    + para.data * (1. - self.m)
        

    def prune_step(self,param_prune_rate:float):

        # global prune encoder_q 
        print('pruning percent:{}'.format(param_prune_rate))
        prune.global_unstructured(
            self.parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=param_prune_rate,
        )

    def add_prune_mask(self):
        # add prune mask for query
        
        for _,m in self.encoder_q.named_modules():
            if isinstance(m,nn.Conv2d):
                prune.random_unstructured(m,'weight',0)
                self.parameters_to_prune.append((m,'weight'))
        for _,m in self.encoder_k.named_modules():
            if isinstance(m,nn.Conv2d):
                prune.random_unstructured(m,'weight',0)
                
        self.parameters_to_prune = tuple(self.parameters_to_prune)



# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class CAMMoCo(MoCoUnstructruedPruned):
    def __init__(self, base_encoder, args, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        super().__init__(base_encoder, args, dim=dim, K=K, m=m, T=T, mlp=mlp)
        self.encoder_k.avg_pool = Identity()
        self.encoder_q.avg_pool = Identity()
        dim_mlp = self.encoder_q.fc[0].weight.shape[1]
        dim_cat = self.encoder_q.fc[2].weight.shape[0]
        self.encoder_q.fc = Identity()
        self.encoder_k.fc = Identity()
        self.encoder_q_head = nn.Sequential(nn.Conv2d(dim_mlp,dim_mlp,1),nn.ReLU(),nn.Conv2d(dim_mlp,dim_cat,1))
        self.encoder_k_head = nn.Sequential(nn.Conv2d(dim_mlp,dim_mlp,1),nn.ReLU(),nn.Conv2d(dim_mlp,dim_cat,1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def load_state_dict_from_MoCo(self,state_dict):
        st = self.state_dict()
        for ky in state_dict:
            if ky[7:] in st:
                #print('load {} in original model para'.format(ky))
                st[ky[7:]] = state_dict[ky]
            else:
                #print(ky)
                if ky == 'module.encoder_q.fc.0.weight' or ky == 'module.encoder_q.fc.2.weight':
                    sp = state_dict[ky].shape
                    st[ky[7:].replace('.fc','_head')] = state_dict[ky].reshape(sp[0],sp[1],1,1)
                elif ky == 'module.encoder_q.fc.0.bias' or ky == 'module.encoder_q.fc.2.bias' :
                    st[ky[7:].replace('.fc','_head')] = state_dict[ky]
                else:
                    print('missing key:',ky)
        
        self.load_state_dict(st)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.encoder_q_head.parameters(), self.encoder_k_head.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
    def _pred_feature_center(self,cam_map):
        B,C,H,W = cam_map.shape
        cam_map = cam_map.permute(0,2,3,1)
        # B,H,W,C
        cam_prob_map = torch.softmax(cam_map,dim=3)
        cam_prob_map = cam_prob_map.view(B,-1,C)

        xind = torch.arange(0, H,step=1,dtype=torch.long)
        yind = torch.arange(0, W,step=1,dtype=torch.long)
        
        grid_x,grid_y = torch.meshgrid([xind,yind])
        grid_x = grid_x.view(1,-1,1)
        grid_y = grid_y.view(1,-1,1)

        mean_x = torch.sum(cam_prob_map*grid_x,dim=1)
        mean_y = torch.sum(cam_prob_map*grid_y,dim=1)

        # mean_{x,y}.shape = B,C
        return mean_x,mean_y

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder, mask value will goes to zero
        """
        for name,para in self.encoder_q.named_parameters():
            if '_orig' in name:
                self.encoder_k.state_dict()[name].data = self.encoder_k.state_dict()[name].data * self.m \
                    + torch.mul(para.data, self.encoder_q.state_dict()[name.replace('_orig','_mask')]) * (1. - self.m)
            else:
                self.encoder_k.state_dict()[name].data = self.encoder_k.state_dict()[name].data * self.m \
                    + para.data * (1. - self.m)
        for param_q, param_k in zip(self.encoder_q_head.parameters(), self.encoder_k_head.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    def forward(self, im_q, im_k,trans_q,trans_k):
        # TODO: check
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            trans_q: shape=(B,4), represent the linear transormations perform on image q,[kh,kw,bh,bw]
            trans_k: shape=(B,4), represent the linear transormations perform on image k,[kh,kw,bh,bw]
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxCxHxW
        q = self.encoder_q_head(q) 
        # TODO: Check if perform avg before norm can help classification
        q = nn.functional.normalize(q,dim=1)
        mean_x_q,mean_y_q = self._pred_feature_center(q)
        q = self.avg_pool(q).squeeze() # q: NxC

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxCxWxH
            k = self.encoder_k_head(k) 
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            
            # TODO: Check if perform avg before norm can help classification
            mean_x_k,mean_y_k = self._pred_feature_center(k)
            k = self.avg_pool(k).squeeze() # q: NxC

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        # calculate the shift difference for pos samples
        l_reg_x = (mean_x_q-trans_q[2])/trans_q[0] - (mean_x_k-trans_k[2])/trans_k[0]
        l_reg_y = (mean_y_q-trans_q[3])/trans_q[1] - (mean_y_k-trans_k[3])/trans_k[1]

        reg_loss = torch.sum(l_reg_y**2 + l_reg_x **2)

        return logits, labels,reg_loss
    

if __name__ == '__main__':
    import torchvision.models as models
    new_model = CAMMoCo(models.resnet50,None,mlp=True)
    cp = torch.load('moco_v2_200ep_pretrain.pth.tar')
    new_model.load_state_dict_from_MoCo(cp['state_dict'])
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import moco.builder
import moco.loader


class CropWithPara(transforms.RandomResizedCrop):
    def forward(self,img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation),i,j,h,w

class HFlipWithPara(transforms.RandomHorizontalFlip):
    def forward(self, img):
        if torch.rand(1) < self.p:
            return F.hflip(img),torch.tensor(1)
        return img,torch.tensor(0)

class AugPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.totensor = transforms.ToTensor()
        self.hflip = HFlipWithPara()
        self.blur = transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5)
        self.grayscale = transforms.RandomGrayscale(p=0.2)
        self.jitter = transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) 
            ], p=0.8)
        self.crop = CropWithPara(224,scale=(0.2,1.))
    
    def _get_trans(self,i,j,h,w,flip):
        kh = 224./h
        bh = -224./h*i
        if flip:
            kw = -224./w
            bw = 223 + (224./w)*j 
        else:
            kw = 224./w
            bw = -224./w * j
        return torch.tensor([kh,kw,bh,bw])

    def forward(self,img):
        # 2 branches of img
        img1,i1,j1,h1,w1 = self.crop(img)
        img1 = self.jitter(img1)
        img1 = self.grayscale(img1)
        img1 = self.blur(img1)
        img1,flip1 = self.hflip(img1)
        img1 = self.totensor(img1)
        img1 = self.normalize(img1)
        
        linear_trans1 = self._get_trans(i1,j1,h1,w1,flip1)

        img2,i2,j2,h2,w2 = self.crop(img)
        img2 = self.jitter(img2)
        img2 = self.grayscale(img2)
        img2 = self.blur(img2)
        img2,flip2 = self.hflip(img2)
        img2 = self.totensor(img2)
        img2 = self.normalize(img2)
        
        linear_trans2 = self._get_trans(i2,j2,h2,w2,flip2)
        return img1,img2,linear_trans1,linear_trans2



        
if __name__ == '__main__':
    alg = AugPlus()
    import torchvision.transforms
    topil = torchvision.transforms.ToPILImage()
    ts = torch.zeros([3,255,255])
    plx = 180
    ply = 127
    ts[0][plx][ply] = 1
    pil = topil(ts)

    aug = alg(pil)
    i1 = aug[0]
    i2 = aug[1]
    print(torch.where(i1>0.1))
    print(torch.where(i2>0.1))
    print(aug[2],aug[3])
        

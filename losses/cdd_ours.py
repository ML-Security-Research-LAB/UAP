import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

def MMD(x, y, kernel='multiscale'):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    device = x.device
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
      

    return torch.mean(XX + YY - 2. * XY)

 

class CDDLossOurs(nn.Module):

    def __init__(self, device='cuda:0'):
        super().__init__()
        self.mmd = MMD
        print('Using CDDLossOurs')

    def forward(self, f, y, means, covs):
        
        loss1 = 0
        loss2 = 0
        classes = np.unique(y.cpu().numpy())
        for k1 in classes:
            for k2 in classes:
                feat = f[y==k1]
                if k1 == k2:
                    mu = means[k1]
                    cov = covs[k1]
                    dist = torch.distributions.MultivariateNormal(mu, cov)
                    loss_sim = self.mmd(feat, dist.sample((feat.shape[0],)))
                    loss1 += loss_sim 
                else:
                    mu = means[k2]
                    cov = covs[k2]
                    dist = torch.distributions.MultivariateNormal(mu, cov)
                    loss_dif = self.mmd(feat, dist.sample((feat.shape[0],)))
                    loss2 += loss_dif
                
        loss1 /= len(classes)
        loss2 /= (len(classes) * max(len(classes)-1,1))
        
        loss = loss1 - loss2
        return loss
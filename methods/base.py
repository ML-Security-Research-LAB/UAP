import copy
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import ConcatDataset, DataLoader


class Base(nn.Module):
    def __init__(self, args):
        super(Base, self).__init__()
        for name in args.__dict__:
            setattr(self,name,getattr(args,name))

        self.test_transform = transforms.Compose([transforms.Resize((self.img_size,self.img_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # print('test transform inside model class:', self.test_transform)

        self.lossMeter = AverageMeter()
        self.accMeter = AverageMeter()
        
        out_dim = args.z_dim
        
        if args.back_bone == 'resnet18':
            net = models.resnet18(pretrained=True)
            net.fc = nn.Linear(net.fc.in_features,out_dim)
        elif args.back_bone == 'resnet50':
            args.z_dim = out_dim = 2048
            net = models.resnet50(pretrained=True)
            net.fc = nn.Linear(net.fc.in_features,out_dim)
        elif args.back_bone == 'densenet121':
            net = models.densenet121(pretrained=True)
            net.classifier = nn.Linear(net.classifier.in_features,out_dim)
        elif args.back_bone == 'vgg11_bn':
            net = models.vgg11_bn(pretrained=True)
            net.classifier[6] = nn.Linear(net.classifier[6].in_features,out_dim)
        elif args.back_bone == 'vgg16_bn':
            net = models.vgg16_bn(pretrained=True)
            net.classifier[6] = nn.Linear(net.classifier[6].in_features,out_dim)
        elif args.back_bone == 'mnist':
            args.num_classes = 10
            args.z_dim = 64
            net = MNISTNetwork()
        elif args.back_bone == 'deit':
            net = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
            net.head = nn.Linear(net.head.in_features,args.z_dim)
        elif args.back_bone == 'vit_b_16':
            net = models.vit_b_16(pretrained=True)
            # ViT has 768 hidden dimensions by default
            net.heads = nn.Linear(net.heads.head.in_features, out_dim)
        elif args.back_bone == 'vit_b_32':
            net = models.vit_b_32(pretrained=True)
            # ViT-B/32 also has 768 hidden dimensions
            net.heads = nn.Linear(net.heads.head.in_features, out_dim)
        elif args.back_bone == 'vit_l_32':
            net = models.vit_l_32(pretrained=True)
            # ViT-L/32 has 1024 hidden dimensions
            net.heads = nn.Linear(net.heads.head.in_features, out_dim)
        else:
            raise NotImplementedError

        self.net = net
        self.cls = nn.Linear(args.z_dim,args.num_classes)

        self.net.to(args.device)
        self.cls.to(args.device)
        self.model = nn.Sequential(self.net,self.cls)

        optim_list = [
                {'params': self.net.parameters(), 'lr': self.lr, 'momentum': 0.9, 'weight_decay': self.weight_decay, 'name': 'net'},
                {'params': self.cls.parameters(), 'lr': self.lr, 'momentum': 0.9, 'weight_decay': self.weight_decay, 'name': 'cls'}
            ]

        if args.optim == 'SGD':
            self.optim = torch.optim.SGD(optim_list, lr=self.lr)
        elif args.optim == 'Adam':
            self.optim = torch.optim.Adam(optim_list, lr=self.lr)
        else:
            raise NotImplementedError

    def state_dict(self):
        state_dict = {'model_state_dict':self.model.state_dict(),
                        'optim_state_dict':self.optim.state_dict()}
        return state_dict

    def load_state_dict(self,state_dict):
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optim.load_state_dict(state_dict['optim_state_dict'])

    def apply_make_batchnorm(self, momentum, track_running_stats):
        self.model.apply(lambda m: self.make_batchnorm(m, momentum, track_running_stats))

    @staticmethod
    def make_batchnorm(m, momentum, track_running_stats):
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum
            m.track_running_stats = track_running_stats
            if track_running_stats:
                m.register_buffer('running_mean', torch.zeros(m.num_features, device=m.weight.device))
                m.register_buffer('running_var', torch.ones(m.num_features, device=m.weight.device))
                m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=m.weight.device))
            else:
                m.running_mean = None
                m.running_var = None
                m.num_batches_tracked = None
        return m

    def make_batchnorm_dataset(self, server_trainloader, client_trainloader, transform):
        batchnorm_dataset = []

        server_dataset = copy.deepcopy(server_trainloader.dataset)
        server_dataset.transform = transform
        batchnorm_dataset.append(server_dataset)
        for i in range(len(client_trainloader)):
            client_dataset = copy.deepcopy(client_trainloader[i].dataset)
            client_dataset.transform = transform
            batchnorm_dataset.append(client_dataset)

        self.batchnorm_dataset = ConcatDataset(batchnorm_dataset)


    @torch.no_grad()
    def make_batchnorm_stats(self, server_trainloader, client_trainloader, transform):
        self.test_model = copy.deepcopy(self.model)
        self.test_model.apply(lambda m: self.make_batchnorm(m, None, True))
        
        if hasattr(self, 'batchnorm_dataset'):
            data_loader = DataLoader(self.batchnorm_dataset, shuffle=False, batch_size=self.batchsize, num_workers=4)
        else:
            self.make_batchnorm_dataset(server_trainloader, client_trainloader, transform)
            data_loader = DataLoader(self.batchnorm_dataset, shuffle=False, batch_size=self.batchsize, num_workers=4)
        
        self.test_model.train(True)
        for batch_idx, data in enumerate(data_loader):
            x,_ = data[0]
            x = x.to(self.device)
            self.test_model(x)

    @staticmethod       
    def make_optimizer(model):
        optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.5,
                            weight_decay=0, nesterov=False)
        return optimizer

    @staticmethod
    def save_optimizer_state_dict(optimizer_state_dict):
        optimizer_state_dict_ = {}
        for k, v in optimizer_state_dict.items():
            if k == 'state':
                state_dict_cpu = {}
                for inner_k, inner_v in v.items():
                    if isinstance(inner_v, dict):
                        inner_v_cpu = {param_k: param_v.cpu() if isinstance(param_v, torch.Tensor) else param_v for param_k, param_v in inner_v.items()}
                        state_dict_cpu[inner_k] = inner_v_cpu
                    else:
                        state_dict_cpu[inner_k] = inner_v.cpu() if isinstance(inner_v, torch.Tensor) else inner_v
                optimizer_state_dict_[k] = state_dict_cpu
            else:
                optimizer_state_dict_[k] = copy.deepcopy(v)
        return optimizer_state_dict_

    def get_lr(self):
        return self.optim.param_groups[0]['lr']

    def freeze_classifier(self):
        for param in self.cls.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self):
        for param in self.cls.parameters():
            param.requires_grad = True

    def ema(self, previous, current, alpha):
        return alpha * previous + (1 - alpha) * current

import functools

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} took {end - start} seconds')
        return result
    return wrapper
        
    
    

class MNISTNetwork(nn.Module):
    def __init__(self):
        super(MNISTNetwork,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.fc11 = nn.Sequential(nn.Linear(1024, 64))

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()


    def forward(self,x):
        h = self.encoder(x)
        h = h.view(-1, 1024)
        z = F.relu(self.fc11(h))
        return z

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.count = 0
        self.sum = 0
    def update(self,val,n=1):
        self.count += n
        self.sum += val*n
    def average(self):
        return self.sum/self.count
    def __repr__(self):
        r = self.sum/self.count
        if r<1e-3:
            return '{:.2e}'.format(r)
        else:
            return '%.4f'%(r)
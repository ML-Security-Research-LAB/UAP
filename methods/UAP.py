import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import torch.autograd as autograd
import torch.distributions as dist
import torch.cuda.amp as amp  # Step 1: Import the necessary module
import numpy as np
from .base import *
from tqdm import tqdm
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from losses import CDDLoss, CovLoss



class Model(Base):
    def __init__(self, args):
        super(Model, self).__init__(args)
        print('Using UAP method')
        print('alpha:', self.alpha, 'beta:', self.beta)


    def initialize_local_training(self):
        self.train()

        if self.mixed_precision:
            self.scaler = amp.GradScaler()  
            
        if not hasattr(self, 'cdd_loss') and not hasattr(self, 'cov_loss'):
            num_classes, feature_size = self.cls.weight.shape
            self.cdd_loss = CDDLoss(num_classes=num_classes, device=self.device)

            self.cov_loss = CovLoss(self.batchsize, self.cls.weight.shape[1], device=self.device)

            # covariance matrix stays the same throughout training (lambda * Identity)
            self.distribution_covs = [self.lambda_ * torch.eye(feature_size).to(self.device) for i in range(num_classes)]

        
    def forward(self, x):
        return self.model(x)
    
    @torch.no_grad()
    def evaluate(self, test_dl, test_model=False):
        self.lossMeter.reset()
        self.accMeter.reset()

        model = self.test_model if test_model else self
        model.eval()

        for data in test_dl:
            x, y = data[0]
            x, y = x.to(self.device), y.to(self.device)
            preds = model(x)
            loss = F.cross_entropy(preds, y)
            acc = (torch.argmax(preds, 1) == y).float().mean()
            self.lossMeter.update(loss.data, x.shape[0])
            self.accMeter.update(acc.data, x.shape[0])
        return self.lossMeter.average().item(), self.accMeter.average().item()
    

    def train_supervised(self,train_dl,test_dl=None):    
        self.initialize_local_training()

            
        pbar = tqdm(range(self.E))
        for epoch in pbar:
            self.lossMeter.reset()
            self.accMeter.reset()

            self.train()
            for b_idx, data in enumerate(train_dl):
                x, y = data[0]
                x, y = x.to(self.device), y.to(self.device)
                
                # Use autocast for mixed precision
                with amp.autocast(enabled=self.mixed_precision):
                    feats = self.net(x)
                    logits = self.cls(feats)
                    
                    cov_loss = self.cov_loss(feats)
                    
                    # use cdd loss only in the last epoch
                    if epoch == self.E - 1:
                        loss = F.cross_entropy(logits, y) + self.alpha * self.cdd_loss(feats, y, self.cls.weight, self.distribution_covs) + self.beta * cov_loss
                    else:
                        loss = F.cross_entropy(logits, y) + self.beta * cov_loss
                    
                    # print(loss, cov_loss)

                self.optim.zero_grad()
                
                if self.mixed_precision:
                    # Use the scaler for backprop
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optim.step()
                    
                pbar.set_postfix({'batch_idx': b_idx, 'total_batches': len(train_dl)})


                acc = (logits.argmax(1)==y).float().mean()
                self.lossMeter.update(loss.data,x.shape[0])
                self.accMeter.update(acc.data,x.shape[0])

            if self.scheduler:
                self.scheduler.step()
            
            pbar.set_postfix({'server_loss': self.lossMeter.average().item(), 'server_acc': self.accMeter.average().item(), \
                              'server_lr': self.get_lr()})


    @timeit
    def get_pseudo_labels(self, loader, epsilon=1e-5, distance='cosine', class_num=7, threshold=0, server=None):
        if server is not None:
            print('Using server to generate pseudo labels...')

        loader = DataLoader(loader.dataset, batch_size=self.batchsize, shuffle=False, num_workers=4)
        
        self.eval()
        start_test = True
        with torch.no_grad():
            # iter_test = iter(loader)
            # idxs = []
            for data in loader:
                inputs, labels = data[0]
                inputs = inputs.to(self.device)
                
                if server is not None:
                    feas = server.net(inputs)
                    outputs = server.cls(feas)
                else:
                    feas = self.net(inputs)
                    outputs = self.cls(feas)

                    
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)

        all_output = nn.Softmax(dim=1)(all_output)
        
        # if not hasattr(self, 'all_output'):
        #     self.all_output = all_output
        # else:
        #     self.all_output = self.ema(self.all_output, all_output, 0.1)
        #     all_output = self.all_output.clone().detach()
        
        
        
        ent = torch.sum(-all_output * torch.log(all_output + epsilon), dim=1)
        unknown_weight = 1 - ent / np.log(class_num)
        _, predict = torch.max(all_output, 1)

        if distance == 'cosine':
            all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        all_fea = all_fea.float().cpu().numpy()
        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()

        for _ in range(2):
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
            cls_count = np.eye(K)[predict].sum(axis=0)
            labelset = np.where(cls_count>threshold)
            labelset = labelset[0]

            dd = cdist(all_fea, initc[labelset], distance)
            pred_label = dd.argmin(axis=1)
            predict = labelset[pred_label]

            aff = np.eye(K)[predict]

        return torch.from_numpy(predict.astype('int')).to(self.device)


       
    def train_unsupervised(self,train_dl,server=None):
        self.initialize_local_training()    

        self.freeze_classifier()

        self.distribution_means = self.cls.weight.data.clone()
        self.distribution_means.detach_()

        pbar = tqdm(range(self.E))
        for epoch in pbar:
            self.lossMeter.reset()
            self.accMeter.reset()

            server = None if epoch > 0 else server
            y_pseudo = self.get_pseudo_labels(train_dl, class_num=self.cls.weight.shape[0], server=server)

            if self.labeled_data > 0:
                print('setting labeled data for {} percent samples'.format(self.labeled_data))
                y_pseudo[train_dl.dataset.sampled_indices] = train_dl.dataset.sampled_indices_labels.to(self.device)

            self.train()
            for batch_idx, data in enumerate(train_dl):
                x, y = data[0]
                idx = data[1]
                x, y = x.to(self.device), y.to(self.device)
                
                with amp.autocast(enabled=self.mixed_precision):
                    feats = self.net(x)
                    pred = self.cls(feats)
                    
                    cov_loss = self.cov_loss(feats)
                    
                    if epoch == self.E - 1:
                        loss = F.cross_entropy(pred, y_pseudo[idx]) + self.alpha * self.cdd_loss(feats, y_pseudo[idx], self.distribution_means, self.distribution_covs) + \
                            self.beta * cov_loss
                    else:
                        loss = F.cross_entropy(pred, y_pseudo[idx]) + self.beta * cov_loss

                    # print(loss, cov_loss)

                self.optim.zero_grad()
                
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optim.step()
                    
                pbar.set_postfix({'batch_idx': batch_idx, 'total_batches': len(train_dl)})

                acc = (pred.argmax(1)==y).float().mean()
                
                self.lossMeter.update(loss.data,x.shape[0])
                self.accMeter.update(acc.data,x.shape[0])
                
            if self.scheduler:
                self.scheduler.step()

            pbar.set_postfix({'client_loss': self.lossMeter.average().item(), 'client_acc': self.accMeter.average().item(),\
                              'client_lr': self.get_lr()})	






        
        
      
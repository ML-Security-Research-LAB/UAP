import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

from .base import *
from tqdm import tqdm
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
import torch.cuda.amp as amp  # Step 1: Import the necessary module


class Model(Base):
    def __init__(self, args):
        super(Model, self).__init__(args)
        print('SSFL method')


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
        
        if self.mixed_precision:
            self.scaler = amp.GradScaler()
        
        # local communication round
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
                    loss = F.cross_entropy(logits, y)

                self.optim.zero_grad()
                
                if self.mixed_precision:
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
            print('using server to generate pseudo labels...')

        # copied_dataset = copy.deepcopy(loader.dataset)
        # copied_dataset.transform = self.test_transform
        loader = DataLoader(loader.dataset, batch_size=self.batchsize, shuffle=False, num_workers=4)
        # loader = DataLoader(copied_dataset, batch_size=self.batchsize, shuffle=False, num_workers=4)
        
        self.eval()

        if server is not None:
            server.eval()

        start_test = True
        with torch.no_grad():
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

        if self.mixed_precision:
            self.scaler = amp.GradScaler()

        self.freeze_classifier()

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

                # Use autocast for mixed precision
                with amp.autocast(enabled=self.mixed_precision):
                    feats = self.net(x)
                    pred = self.cls(feats)

                    loss = F.cross_entropy(pred, y_pseudo[idx])

                self.optim.zero_grad()

                # If using mixed precision, scale the loss, perform backward pass, and unscale
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optim.step()
                

                acc = (pred.argmax(1) == y).float().mean()
                self.lossMeter.update(loss.data, x.shape[0])
                self.accMeter.update(acc.data, x.shape[0])
                
                pbar.set_postfix({'batch_idx': batch_idx, 'total batches': len(train_dl)})
            
            if self.scheduler:
                self.scheduler.step()

            pbar.set_postfix({'client_loss': self.lossMeter.average().item(), 'client_acc': self.accMeter.average().item(), \
                              'client_lr': self.get_lr()})





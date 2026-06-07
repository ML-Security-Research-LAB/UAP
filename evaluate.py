import os 
import random
import sys, warnings
warnings.filterwarnings('ignore') 
import importlib
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from datetime import datetime
import subprocess
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd

from config import get_args
from datasets import *
from losses import *
from utils import *

args = get_args(print_args=True)
data = eval(args.dataset)(root=args.dataset_folder,test_envs=[args.test_env])
args.num_classes = data.num_classes


method = args.method
dataset = args.dataset
test_domain = args.test_env

for domain in range(4):
    if domain == test_domain:
        continue

    # Update server_domain to match the checkpoint being evaluated
    args.server_domain = domain

    server = importlib.import_module('methods.'+args.method).Model(args=args).to(args.device)
    server_trainloader, client_trainloader, target_testloader = get_dataloaders(args, data)

    print(f'\n=== Server Domain {domain} ===')
    weight_path = f'results/{args.experiment_path}/{dataset}/{dataset}/{method}/target_{test_domain}/server_{domain}/seed_0/checkpoint.pt'
    csv_path = f'results/{args.experiment_path}/{dataset}/{dataset}/{method}/target_{test_domain}/server_{domain}/seed_0/target_accs.csv'

    print(f'Loading weights from: {weight_path}')
    checkpoint = torch.load(weight_path, map_location='cpu')

    # Debug: Check weights before and after loading
    before_sum = next(server.model.parameters()).sum().item()
    expected_sum = list(checkpoint['server_state_dict']['model_state_dict'].values())[0].sum().item()

    # Only load model weights, not optimizer (optimizer state can cause issues when loading multiple checkpoints)
    server.model.load_state_dict(checkpoint['server_state_dict']['model_state_dict'])

    after_sum = next(server.model.parameters()).sum().item()
    print(f'Weight loading: before={before_sum:.4f}, after={after_sum:.4f}, expected={expected_sum:.4f}, match={abs(after_sum - expected_sum) < 1e-3}')
    server.eval()

    # Clear any cached test_model to ensure fresh batchnorm stats
    if hasattr(server, 'test_model'):
        delattr(server, 'test_model')
    if hasattr(server, 'batchnorm_dataset'):
        delattr(server, 'batchnorm_dataset')

    # Recalculate batchnorm statistics before evaluation (same as in training)
    server.make_batchnorm_stats(server_trainloader, client_trainloader, target_testloader.dataset.transform)
    target_loss, target_acc = server.evaluate(target_testloader, test_model=True)

    df = pd.read_csv(csv_path)

    print(f'CSV reported acc: {df.target_acc.values[-1]:.4f}')
    print(f'Eval acc: {target_acc:.4f}')
    print(f'Difference: {abs(df.target_acc.values[-1] - target_acc):.4f}')


     

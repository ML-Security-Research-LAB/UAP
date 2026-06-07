import os
import copy
import torch
import wandb
import random
import methods
import numpy as np
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from mnist_datasets import get_dataloaders as get_mnist_dataloaders


def SEED_EVERYTHING(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  # Use deterministic algorithms
    torch.backends.cudnn.benchmark = False     # Disable benchmark for reproducibility


def initialize_wandb(args, project_name, args_dict):
    wandb.init(
        project=project_name,
        name=f"{args.method}_{args.ext}_{args.seed}",
        config=args_dict
    )
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py") and 'wandb' not in root:
                wandb.save(os.path.join(root, file))

def load_model(args):
    if args.method == 'SSFL':
        model = methods.SSFL(args=args).to(args.device)
    elif args.method == 'UAP':
        model = methods.UAP(args=args).to(args.device)
    else:
        raise ValueError(f"Unknown method: {args.method}. Available methods: SSFL, UAP")
    return model

def set_client_models(args, client_trainloader):
    return [load_model(args) for _ in range(len(client_trainloader))]

def log_wandb_metrics(args, target_acc, remaining_estimated_time):
    if args.wandb:
        wandb.log({"target_acc": target_acc, "remaining_estimated_time": round(remaining_estimated_time/3600, 2)})
        
def add_labeled_data(dataset, args):
    if args.labeled_data == 0:
        return dataset
    # Group indices by class
    class_indices = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx][0]
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # Shuffle indices within each class
    for label in class_indices:
        np.random.shuffle(class_indices[label])

    # Sample indices per class according to labeled_data percentage
    labeled_percentage = args.labeled_data / 100.0  # Convert percentage to decimal
    sampled_indices = []
    indices_labels = []  # Store labels corresponding to indices

    for label, indices in class_indices.items():
        # Calculate how many samples to take from this class
        num_samples = int(len(indices) * labeled_percentage)
        print(f"Class {label}: Selecting {num_samples} samples out of {len(indices)}")
        
        # Ensure at least one sample per class if labeled_percentage > 0
        if labeled_percentage > 0 and num_samples == 0:
            num_samples = 1
        
        # Take the first num_samples indices
        class_sampled_indices = indices[:num_samples]
        sampled_indices.extend(class_sampled_indices)
        
        # Store corresponding labels
        indices_labels.extend([label] * len(class_sampled_indices))

    # Convert to numpy arrays if needed
    sampled_indices = torch.tensor(sampled_indices)
    indices_labels = torch.tensor(indices_labels)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of sampled examples: {len(sampled_indices)} ({args.labeled_data}% of data)")
    print(f"Sampled indices shape: {sampled_indices.shape}")
    print(f"Indices labels shape: {indices_labels.shape}")
    
    # Make sure sampled_indices and their labels are attributes of the dataset
    dataset.sampled_indices = sampled_indices
    dataset.sampled_indices_labels = indices_labels
    return dataset
            

def get_dataloaders(args, data):
    
    if args.dataset.lower() == 'mnist':
        server_trainloader, client_trainloader, testloader = get_mnist_dataloaders(args.server_domain, args.test_env, 
                                                                                   args.batchsize)
        return server_trainloader, client_trainloader, testloader
    
    client_trainloader = []
    for c_id in range(len(data.ENVIRONMENTS)):
        if c_id == args.server_domain:
            dataset = data.datasets[c_id]
            print('server labeled domain', c_id, dataset.root if args.dataset != 'WILDSCamelyon' else dataset.name)
            # print(dataset.transform)
            server_trainloader = DataLoader(dataset,batch_size=args.batchsize,
                                           shuffle=True,num_workers=4)

            
        elif c_id == args.test_env:
            testset = data.datasets[args.test_env]
            print('test domain', c_id, testset.root if args.dataset != 'WILDSCamelyon' else testset.name)
            # print(testset.transform)
            testloader = DataLoader(testset,batch_size=args.batchsize,
                                           shuffle=False,num_workers=4)

        
        else:
            dataset = data.datasets[c_id]
            print('client unlabeled domain', c_id, dataset.root if args.dataset != 'WILDSCamelyon' else dataset.name)

            dataset.transform = data.transform
            # print(dataset.transform)
            dataset = add_labeled_data(dataset, args)

            trainloader = DataLoader(dataset,batch_size=args.batchsize,
                                     shuffle=True,num_workers=4)

            client_trainloader.append(trainloader)
            
   
    return server_trainloader, client_trainloader, testloader


def set_total_steps(args, server, clients, server_dl, clients_dl):
    for i in range(len(clients_dl)):
        # clients[i].total_steps = len(clients_dl[i]) * args.E * args.rounds
        clients[i].total_steps = args.E * args.rounds
        
    # server.total_steps = len(server_dl) * args.server_epochs
    server.total_steps = args.E * args.rounds

def set_scheduler(args, server, clients, server_dl, clients_dl):
    
    if args.lr_scheduler:
        set_total_steps(args, server, clients, server_dl, clients_dl)
        if args.scheduler_type == 'cosine':
            server.scheduler = CosineAnnealingLR(server.optim, T_max=server.total_steps, eta_min=args.min_lr)
            for i in range(len(clients)):
                clients[i].scheduler = CosineAnnealingLR(clients[i].optim, T_max=clients[i].total_steps, eta_min=args.min_lr)
            
            
        elif args.scheduler_type == 'step':
            server.scheduler = StepLR(server.optim, step_size=args.lr_step_size, gamma=args.lr_gamma)
            for i in range(len(clients)):
                clients[i].scheduler = StepLR(clients[i].optim, step_size=args.lr_step_size, gamma=args.lr_gamma)
        else:
            raise NotImplementedError
        
        if args.verbose:
            server.verbose = True
            for i in range(len(clients)):
                clients[i].verbose = True
    else:
        server.scheduler = None
        for i in range(len(clients)):
            clients[i].scheduler = None

@torch.no_grad()
def calculate_client_weights(client_trainloaders):
    # Calculate the number of samples in each client
    N = [len(client_trainloader.dataset) for client_trainloader in client_trainloaders]
    weights = [n / sum(N) for n in N]
    print(weights)
    return weights

@torch.no_grad()
def fedavg_clients(server, clients, skip_bn=False, client_weights=None, uniform_weights=False):
    state_dict = copy.deepcopy(server.model.state_dict())
    client_state_dicts = [client.model.state_dict() for client in clients]
    N = len(clients)
    
    if uniform_weights:
        client_weights = [1.0 / N for _ in clients]
    
    for key in state_dict.keys():
        
        if skip_bn and 'bn' in key:
            continue
        
        
        p = []
        for i in range(len(clients)):
            p.append(client_state_dicts[i][key] * client_weights[i])
            
        
        p_avg = sum(p) 
        state_dict[key] = p_avg
    
    # Update the server's model
    server.model.load_state_dict(state_dict)

def write_results(args, target):
    path = f'{args.experiment_path}/{args.dataset}/{args.method}/target_{args.test_env}/server_{args.server_domain}/seed_{args.seed}'
    os.makedirs(path, exist_ok=True)

    dict_ = {'target_acc':target}
    df = pd.DataFrame(dict_)
    df.to_csv(os.path.join(path, 'target_accs.csv'),index=False)


def save_checkpoint(args, server, clients, hist):
    path = f'{args.experiment_path}/{args.dataset}/{args.method}/target_{args.test_env}/server_{args.server_domain}/seed_{args.seed}'
    os.makedirs(path, exist_ok=True)

    save_dict = {'server_state_dict':server.state_dict()}
    
    for i in range(len(clients)):
        save_dict[f'client_{i}_state_dict'] = clients[i].state_dict()

    save_dict['hist'] = hist
     
    torch.save(save_dict, os.path.join(path,'checkpoint.pt'))









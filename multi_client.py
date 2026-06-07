import numpy as np
from numpy.random import dirichlet
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from mnist_datasets import get_dataloaders as get_mnist_dataloaders

class ClientDatasetWithFixedIndex(Dataset):
    """
    A dataset wrapper that returns each sample along with its fixed local index.
    This index remains the same regardless of shuffling.
    """
    def __init__(self, dataset):
        self.dataset = dataset  # Underlying dataset.

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Assume the underlying dataset returns (x, y) (or ((x, y), ...)).
        # We return x, y along with the fixed index idx.
        item = self.dataset[idx]
        if isinstance(item, tuple) and len(item) >= 1:
            # If the underlying dataset returns ((x, y), something)
            data = item[0] if isinstance(item[0], (list, tuple)) else item
        else:
            data = item
        return data, idx  # returns (x, y, idx)

def get_dataloaders_multi_client(args, data):
    if args.dataset.lower() == 'mnist':
        server_trainloader, client_trainloader, testloader = get_mnist_dataloaders(
            args.server_domain, args.test_env, args.batchsize
        )
        return server_trainloader, client_trainloader, testloader

    # Determine client environment indices (exclude server and test domains)
    client_env_indices = [
        i for i in range(len(data.ENVIRONMENTS))
        if i != args.server_domain and i != args.test_env
    ]

    # Setup server data
    server_dataset = data.datasets[args.server_domain]
    print('server labeled domain', args.server_domain,
          server_dataset.root if args.dataset != 'WILDSCamelyon' else server_dataset.name)
    server_trainloader = DataLoader(
        server_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4
    )

    # Setup test data
    testset = data.datasets[args.test_env]
    print('test domain', args.test_env,
          testset.root if args.dataset != 'WILDSCamelyon' else testset.name)
    testloader = DataLoader(
        testset, batch_size=args.batchsize, shuffle=False, num_workers=4
    )

    # Setup client data: default to 30 clients if not specified
    N = getattr(args, 'num_clients', 30)

    # Create client dataloaders based on the distribution type
    if args.data_distribution == 'single-domain':
        client_trainloaders = _create_single_domain_loaders(args, data, client_env_indices, N)
    elif args.data_distribution == 'multi-domain':
        client_trainloaders = _create_multi_domain_loaders(args, data, client_env_indices, N)
    else:
        raise ValueError(f"Unsupported data_distribution type: {args.data_distribution}")

    return server_trainloader, client_trainloaders, testloader

def _create_single_domain_loaders(args, data, client_env_indices, N):
    """
    Create dataloaders where each client receives data from a single domain.
    Data from each source domain is split in a class-balanced manner among a subset of clients.
    Each client's dataset is wrapped with ClientDatasetWithFixedIndex so that each sample
    returns (x, y, fixed_index) which remains consistent for pseudo-label lookup.
    """
    client_trainloaders = []
    num_domains = len(client_env_indices)
    
    # Determine the number of clients per domain.
    clients_per_domain = [N // num_domains] * num_domains
    for i in range(N % num_domains):
        clients_per_domain[i] += 1
    print(f'Clients per domain: {clients_per_domain}')
    
    # Loop over each source domain.
    for d_idx, env_idx in enumerate(client_env_indices):
        dataset = data.datasets[env_idx]
        print(f'client unlabeled domain {env_idx}',
              dataset.root if args.dataset != 'WILDSCamelyon' else dataset.name)
        
        dataset.transform = data.transform
        
        # Group indices by class
        class_indices = {}
        for idx in range(len(dataset)):
            (_, label), _ = dataset[idx]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # Shuffle indices within each class
        # for label in class_indices:
        #     np.random.shuffle(class_indices[label])
        
        # Distribute class-balanced data to each client
        clients_data = [[] for _ in range(clients_per_domain[d_idx])]
        
        for label, indices in class_indices.items():
            # Calculate samples per client for this class
            samples_per_client = len(indices) // clients_per_domain[d_idx]
            remainder = len(indices) % clients_per_domain[d_idx]
            
            start_idx = 0
            for c in range(clients_per_domain[d_idx]):
                # Assign slightly more samples to early clients if division isn't even
                extra = 1 if c < remainder else 0
                end_idx = start_idx + samples_per_client + extra
                
                # Add these class samples to the client's dataset
                clients_data[c].extend(indices[start_idx:end_idx])
                start_idx = end_idx
        
        # Create loaders for each client from this domain
        for c in range(clients_per_domain[d_idx]):
            # Create a subset and wrap it so that each sample returns a fixed index
            subset = Subset(dataset, clients_data[c])
            client_dataset = ClientDatasetWithFixedIndex(subset)
            
            if True:
                label_distribution = {}
                for idx in range(len(client_dataset)):
                    _, label = client_dataset[idx][0]
                    if label not in label_distribution:
                        label_distribution[label] = 0
                    label_distribution[label] += 1
                print(f"Client {c} has label distribution: {label_distribution} in domain {env_idx}")
            
            loader = DataLoader(
                client_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4
            )
            client_trainloaders.append(loader)
    
    return client_trainloaders

def _create_multi_domain_loaders(args, data, client_env_indices, N):
    """
    Create dataloaders where each client can receive data from multiple domains.
    For each domain, samples are allocated to clients based on a Dirichlet distribution.
    Each client’s subset for each domain is wrapped with ClientDatasetWithFixedIndex,
    ensuring that the fixed local index is maintained.
    """
    # Create an empty list for each client to store subsets from different domains.
    client_datasets = [[] for _ in range(N)]
    alpha = getattr(args, 'dirichlet_alpha', 0.5)
    
    for env_idx in client_env_indices:
        dataset = data.datasets[env_idx]
        print(f'client unlabeled domain {env_idx}',
              dataset.root if args.dataset != 'WILDSCamelyon' else dataset.name)
        
        dataset.transform = data.transform
        num_samples = len(dataset)
        
        # Sample proportions from a Dirichlet distribution for client-wise allocation.
        proportions = dirichlet([alpha] * N)
        client_sample_counts = (proportions * num_samples).astype(int)
        
        # Adjust for rounding errors so that all samples are allocated.
        remaining = num_samples - client_sample_counts.sum()
        client_sample_counts[0] += remaining
        
        start_idx = 0
        # Allocate samples to each client.
        for client_idx in range(N):
            count = client_sample_counts[client_idx]
            if count > 0:
                end_idx = start_idx + count
                indices = list(range(start_idx, end_idx))
                subset = Subset(dataset, indices)
                client_dataset = ClientDatasetWithFixedIndex(subset)
                client_datasets[client_idx].append(client_dataset)
                start_idx = end_idx
    
    # For each client, concatenate all their domain-specific subsets.
    client_trainloaders = []
    for ds_list in client_datasets:
        if ds_list:
            combined_dataset = ConcatDataset(ds_list)
            loader = DataLoader(
                combined_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4
            )
            client_trainloaders.append(loader)
    
    return client_trainloaders

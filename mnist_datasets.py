import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

def get_domainwise_data(num_supervised=1000):
    sup_inds = np.load('./data/roatedmnist_sup_inds/supervised_inds_1.npy')
    # print(sup_inds.shape)

    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/',
                                                              train=True,
                                                              download=True,
                                                              transform=transforms.ToTensor()),
                                                        batch_size=60000,
                                                        shuffle=False)

    for i, (x, y) in enumerate(train_loader):
        mnist_imgs = x
        mnist_labels = y

    # Get num_supervised number of labeled examples
    mnist_labels = mnist_labels[sup_inds]
    mnist_imgs = mnist_imgs[sup_inds]

    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    # Run transforms
    mnist_0_img = torch.zeros((num_supervised, 28, 28))
    mnist_15_img = torch.zeros((num_supervised, 28, 28))
    mnist_30_img = torch.zeros((num_supervised, 28, 28))
    mnist_45_img = torch.zeros((num_supervised, 28, 28))
    mnist_60_img = torch.zeros((num_supervised, 28, 28))
    mnist_75_img = torch.zeros((num_supervised, 28, 28))


    for i in range(len(mnist_imgs)):
        mnist_0_img[i] = to_tensor(to_pil(mnist_imgs[i]))

    for i in range(len(mnist_imgs)):
        mnist_15_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 15))

    for i in range(len(mnist_imgs)):
        mnist_30_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 30))

    for i in range(len(mnist_imgs)):
        mnist_45_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 45))

    for i in range(len(mnist_imgs)):
        mnist_60_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 60))

    for i in range(len(mnist_imgs)):
        mnist_75_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 75))

        
    return mnist_0_img, mnist_15_img, mnist_30_img, mnist_45_img, mnist_60_img, mnist_75_img, mnist_labels


class CustomTensorDataset(Dataset):
    """TensorDataset with support for transforms."""
    def __init__(self, tensors, labels, transform=None):
        assert tensors.size(0) == labels.size(0)  # Make sure the number of samples matches the number of labels
        self.tensors = tensors
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.tensors.size(0)

    def __getitem__(self, index):
        x = self.tensors[index].unsqueeze(0)
        y = self.labels[index]

        if self.transform:
            x = self.transform(x)

        return (x, y), index



def get_dataloaders(server_domain, test_domain, batch_size=64):
    *domains,labels = get_domainwise_data()
    print('total number of domains:', len(domains))
    
    # Get the server domain
    server_dataset = CustomTensorDataset(domains[server_domain], labels)
    server_dataloader = DataLoader(server_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CustomTensorDataset(domains[test_domain], labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    client_dataloaders = []
    for i in range(len(domains)):
        if i == server_domain or i == test_domain:
            continue
        
        client_dataset = CustomTensorDataset(domains[i], labels)
        client_dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        client_dataloaders.append(client_dataloader)
    return server_dataloader, client_dataloaders, test_dataloader




if __name__ == '__main__':
    # mnist_0_img, mnist_15_img, mnist_30_img, mnist_45_img, mnist_60_img, mnist_75_img, mnist_labels = get_domainwise_data()
    # print(mnist_0_img.shape)
    # print(mnist_15_img.shape)
    # print(mnist_30_img.shape)
    # print(mnist_45_img.shape)
    # print(mnist_60_img.shape)
    # print(mnist_75_img.shape)
    # print(mnist_labels.shape)
    get_domain_gen_dataloaders(0, 1)



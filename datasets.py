import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
# from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
import numpy as np

#from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
#from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    "PACS",
    "OfficeHome",
    "VLCS",
    "TerraIncognita",
    "WILDSCamelyon",
]

class MultipleDomainDataset:
    EPOCHS = 100             # Default, if train with epochs, check performance every epoch.
    N_WORKERS = 4            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override
    
    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class MyImageFolder(ImageFolder):
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index), index

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, img_size=224):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        self.transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = self.augment_transform
            else:
                env_transform = self.transform

            path = os.path.join(root, environment)
            env_dataset = MyImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)



class PACS(MultipleEnvironmentImageFolder):
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, augment=True, img_size=224):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, augment, img_size)
        

class OfficeHome(MultipleEnvironmentImageFolder):
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, augment=True, img_size=224):
        self.dir = os.path.join(root, "officehome/")
        super().__init__(self.dir, test_envs, augment, img_size)


class VLCS(MultipleEnvironmentImageFolder):
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, augment=True, img_size=224):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, augment=True, img_size=img_size)

class TerraIncognita(MultipleEnvironmentImageFolder):
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, augment=True, img_size=224):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, augment=True, img_size=img_size)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        
        # print(x)
        # print(y)
        return (x, y), i

    def __len__(self):
        return len(self.indices)

class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 96, 96)
    def __init__(self, dataset, metadata_name, test_envs, augment):
        super().__init__()

        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.augment_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.RandomResizedCrop(96, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = self.augment_transform
            else:
                env_transform = self.transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 96, 96,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, augment=True, img_size=96):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, augment)
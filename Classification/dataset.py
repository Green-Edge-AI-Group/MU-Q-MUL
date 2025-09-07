"""
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100   
"""
import copy
import glob
import os
from shutil import move

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder,EuroSAT
from tqdm import tqdm

import os
import cv2
import pandas as pd
from torch.utils.data import  random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split


def cifar10_dataloaders_no_val(
    batch_size=128, data_dir="datasets/cifar10", num_workers=2
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
    val_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def svhn_dataloaders(
    batch_size=128,
    data_dir="datasets/svhn",
    num_workers=2,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
):
    # train_transform = transforms.Compose(
    #     [

    #         transforms.ToTensor(),
    #     ]
    # )

    # test_transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #     ]
    # )
    mean = [0.4377, 0.4438, 0.4728]
    std = [0.1980, 0.2010, 0.1970]
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    print(
        "Dataset information: SVHN\t 45000 images for training \t 5000 images for validation\t"
    )

    train_set = SVHN(data_dir, split="train", transform=train_transform, download=True)

    test_set = SVHN(data_dir, split="test", transform=test_transform, download=True)

    train_set.labels = np.array(train_set.labels)
    test_set.labels = np.array(test_set.labels)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.labels) + 1):
        class_idx = np.where(train_set.labels == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.labels = train_set_copy.labels[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.labels = train_set_copy.labels[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == 4454:
            test_set.data = test_set.data[test_set.labels != class_to_replace]
            test_set.labels = test_set.labels[test_set.labels != class_to_replace]

    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def cifar100_dataloaders(
    batch_size=128,
    data_dir="datasets/cifar100",
    num_workers=2,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
):
    
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    
    train_transform=transforms.Compose([transforms.Resize(32),transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    test_transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")
    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)

    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None or indexes_to_replace == 450:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader

def svhn_dataloaders(
    batch_size=128,
    data_dir="datasets/svhn",
    num_workers=2,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
):
    # train_transform = transforms.Compose(
    #     [

    #         transforms.ToTensor(),
    #     ]
    # )

    # test_transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #     ]
    # )
    mean = [0.4377, 0.4438, 0.4728]
    std = [0.1980, 0.2010, 0.1970]
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    print(
        "Dataset information: SVHN\t 45000 images for training \t 5000 images for validation\t"
    )

    train_set = SVHN(data_dir, split="train", transform=train_transform, download=True)

    test_set = SVHN(data_dir, split="test", transform=test_transform, download=True)

    train_set.labels = np.array(train_set.labels)
    test_set.labels = np.array(test_set.labels)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.labels) + 1):
        class_idx = np.where(train_set.labels == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.labels = train_set_copy.labels[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.labels = train_set_copy.labels[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == 4454:
            test_set.data = test_set.data[test_set.labels != class_to_replace]
            test_set.labels = test_set.labels[test_set.labels != class_to_replace]

    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):  # 新增长度获取方法
        return len(self.subset)

def eurosat_dataloaders(
    batch_size=128,
    data_dir="datasets/eurosat",
    num_workers=2,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
):
    # Normalization parameters (adjust if needed)
    mean = [0.485, 0.456, 0.406]  # ImageNet defaults
    std = [0.229, 0.224, 0.225]

    base_size = 32
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(base_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # 正确的位置：PIL -> Tensor
            transforms.Normalize(mean, std)
        ] if not no_aug else [
            transforms.Resize(base_size),
            transforms.CenterCrop(base_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    test_transform = transforms.Compose([
        transforms.Resize(base_size),
        transforms.CenterCrop(base_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # Load full dataset
    full_set = EuroSAT(data_dir, transform=None, download=True)
    targets = np.array([full_set[i][1] for i in range(len(full_set))])

    # Split into train+val (80%) and test (20%)
    train_val_idx, test_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        stratify=targets,
        random_state=seed
    )
    
    # Split train_val into train and validation (stratified)
    train_val_targets = targets[train_val_idx]
    rng = np.random.RandomState(seed)
    valid_idx = []
    for i in range(np.max(targets) + 1):
        class_mask = (train_val_targets == i)
        class_indices = np.where(class_mask)[0]
        if len(class_indices) == 0:
            continue
        selected = rng.choice(class_indices, 
                            int(0.1 * len(class_indices)), 
                            replace=False)
        valid_idx.extend(train_val_idx[selected])
    valid_idx = np.array(valid_idx)
    train_idx = np.setdiff1d(train_val_idx, valid_idx)

    # Create datasets with proper transforms
    # 创建子集时直接应用正确的transform
    train_subset = TransformSubset(
        Subset(full_set, train_idx), 
        train_transform
    )
    valid_subset = TransformSubset(
        Subset(full_set, valid_idx),
        test_transform
    )
    test_subset = TransformSubset(
        Subset(full_set, test_idx),
        test_transform
    )

    # Add targets for data replacement logic
    train_subset.targets = targets[train_idx]
    valid_subset.targets = targets[valid_idx]
    test_subset.targets = targets[test_idx]

    # Data replacement functionality
    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError("Cannot specify both class_to_replace and indexes_to_replace")
    
    if class_to_replace is not None:
        replace_class(train_subset, class_to_replace, 
                     num_indexes_to_replace=num_indexes_to_replace,
                     seed=seed-1, only_mark=only_mark)
        if num_indexes_to_replace is None:
            # Filter test set
            mask = test_subset.targets != class_to_replace
            test_subset = Subset(test_subset, np.where(mask)[0])
            test_subset.targets = test_subset.targets[mask]

    if indexes_to_replace is not None:
        replace_indexes(train_subset, indexes=indexes_to_replace,
                       seed=seed-1, only_mark=only_mark)

    # Create dataloaders
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
    }

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_subset,
        shuffle=shuffle,
        worker_init_fn=_init_fn if seed else None,
        **loader_args
    )
    val_loader = DataLoader(
        valid_subset,
        shuffle=False,
        worker_init_fn=_init_fn if seed else None,
        **loader_args
    )
    test_loader = DataLoader(
        test_subset,
        shuffle=False,
        worker_init_fn=_init_fn if seed else None,
        **loader_args
    )

    print(f"EuroSAT Dataset: {len(train_subset)} train, {len(valid_subset)} val, {len(test_subset)} test")
    return train_loader, val_loader, test_loader

def cifar100_dataloaders_no_val(
    batch_size=128, data_dir="datasets/cifar100", num_workers=2
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
    val_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class TinyImageNetDataset(Dataset):
    def __init__(self, image_folder_set, norm_trans=None, start=0, end=-1):
        self.imgs = []
        self.targets = []
        self.transform = image_folder_set.transform
        for sample in tqdm(image_folder_set.imgs[start:end]):
            self.targets.append(sample[1])
            img = transforms.ToTensor()(Image.open(sample[0]).convert("RGB"))
            if norm_trans is not None:
                img = norm_trans(img)
            self.imgs.append(img)
        self.imgs = torch.stack(self.imgs)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.imgs[idx]), self.targets[idx]
        else:
            return self.imgs[idx], self.targets[idx]


class TinyImageNet:
    """
    TinyImageNet dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = (
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if normalize
            else None
        )

        self.tr_train = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        self.tr_test = [transforms.Resize(224),]

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

        self.train_path = os.path.join(args.data_dir, "train/")
        self.val_path = os.path.join(args.data_dir, "val/")
        self.test_path = os.path.join(args.data_dir, "test/")

        if os.path.exists(os.path.join(self.val_path, "images")):
            if os.path.exists(self.test_path):
                os.rename(self.test_path, os.path.join(args.data_dir, "test_original"))
                os.mkdir(self.test_path)
            val_dict = {}
            val_anno_path = os.path.join(self.val_path, "val_annotations.txt")
            with open(val_anno_path, "r") as f:
                for line in f.readlines():
                    split_line = line.split("\t")
                    val_dict[split_line[0]] = split_line[1]

            paths = glob.glob(os.path.join(args.data_dir, "val/images/*"))
            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if not os.path.exists(self.val_path + str(folder)):
                    os.mkdir(self.val_path + str(folder))
                    os.mkdir(self.val_path + str(folder) + "/images")
                if not os.path.exists(self.test_path + str(folder)):
                    os.mkdir(self.test_path + str(folder))
                    os.mkdir(self.test_path + str(folder) + "/images")

            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if len(glob.glob(self.val_path + str(folder) + "/images/*")) < 25:
                    dest = self.val_path + str(folder) + "/images/" + str(file)
                else:
                    dest = self.test_path + str(folder) + "/images/" + str(file)
                move(path, dest)

            os.rmdir(os.path.join(self.val_path, "images"))

    def data_loaders(
        self,
        batch_size=128,
        data_dir="datasets/tiny",
        num_workers=2,
        class_to_replace: int = None,
        num_indexes_to_replace=None,
        indexes_to_replace=None,
        seed: int = 1,
        only_mark: bool = False,
        shuffle=True,
        no_aug=False,
    ):
        train_set = ImageFolder(self.train_path, transform=self.tr_train)
        train_set = TinyImageNetDataset(train_set, self.norm_layer)
        test_set = ImageFolder(self.test_path, transform=self.tr_test)
        test_set = TinyImageNetDataset(test_set, self.norm_layer)
        train_set.targets = np.array(train_set.targets)
        train_set.targets = np.array(train_set.targets)
        rng = np.random.RandomState(seed)
        valid_set = copy.deepcopy(train_set)
        valid_idx = []
        for i in range(max(train_set.targets) + 1):
            class_idx = np.where(train_set.targets == i)[0]
            valid_idx.append(
                rng.choice(class_idx, int(0.0 * len(class_idx)), replace=False)
            )
        valid_idx = np.hstack(valid_idx)
        train_set_copy = copy.deepcopy(train_set)

        valid_set.imgs = train_set_copy.imgs[valid_idx]
        valid_set.targets = train_set_copy.targets[valid_idx]

        train_idx = list(set(range(len(train_set))) - set(valid_idx))

        train_set.imgs = train_set_copy.imgs[train_idx]
        train_set.targets = train_set_copy.targets[train_idx]

        if class_to_replace is not None and indexes_to_replace is not None:
            raise ValueError(
                "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
            )
        if class_to_replace is not None:
            replace_class(
                train_set,
                class_to_replace,
                num_indexes_to_replace=num_indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )
            if num_indexes_to_replace is None or num_indexes_to_replace == 500:
                test_set.targets = np.array(test_set.targets)
                test_set.imgs = test_set.imgs[test_set.targets != class_to_replace]
                test_set.targets = test_set.targets[
                    test_set.targets != class_to_replace
                ]
                test_set.targets = test_set.targets.tolist()
        if indexes_to_replace is not None:
            replace_indexes(
                dataset=train_set,
                indexes=indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )

        loader_args = {"num_workers": 0, "pin_memory": False}

        def _init_fn(worker_id):
            np.random.seed(int(seed))

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        val_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, val_loader, test_loader
    



class ImageNetDataset(Dataset):
    def __init__(self, image_folder_set, norm_trans=None):
        self.transform = image_folder_set.transform
        self.img_paths = np.array([sample[0] for sample in image_folder_set.imgs])
        self.targets = np.array([sample[1] for sample in image_folder_set.imgs])
        self.norm_trans = norm_trans

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        
        img = transforms.ToTensor()(img)
        
        if self.norm_trans is not None:
            img = self.norm_trans(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, self.targets[idx]

class ImageNet:
    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = (
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if normalize
            else None
        )

        self.tr_train = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        self.tr_test = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

        self.train_path = os.path.join("/home/data_SSD/tyj/dataset/imagenet","train")
        self.test_path = os.path.join("/home/data_SSD/tyj/dataset/imagenet", "val")

    # def data_loaders(
    #     self,
    #     batch_size=256,
    #     num_workers=4,
    #     seed: int = 1,
    #     shuffle=True,
    #     no_aug=False,
    # ):
    #     train_set = ImageFolder(self.train_path, transform=self.tr_train)
    #     train_set = ImageNetDataset(train_set, self.norm_layer)
    #     val_set = ImageFolder(self.val_path, transform=self.tr_val)
    #     val_set = ImageNetDataset(val_set, self.norm_layer)

    #     loader_args = {"num_workers": num_workers, "pin_memory": True}

    #     def _init_fn(worker_id):
    #         np.random.seed(int(seed))

    #     train_loader = DataLoader(
    #         train_set,
    #         batch_size=batch_size,
    #         shuffle=shuffle,
    #         worker_init_fn=_init_fn if seed is not None else None,
    #         **loader_args,
    #     )
    #     val_loader = DataLoader(
    #         val_set,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         worker_init_fn=_init_fn if seed is not None else None,
    #         **loader_args,
    #     )

    #     print(
    #         f"Training loader: {len(train_loader.dataset)} images, Validation loader: {len(val_loader.dataset)} images"
    #     )
    #     return train_loader, val_loader

    def data_loaders(
        self,
        batch_size=128,
        data_dir="datasets/imagenet",
        num_workers=2,
        class_to_replace: int = None,
        num_indexes_to_replace=None,
        indexes_to_replace=None,
        seed: int = 1,
        only_mark: bool = False,
        num_classes_to_replace=None,
        shuffle=True,
        no_aug=False,
    ):
        train_set = ImageFolder(self.train_path, transform=self.tr_train)
        train_set = ImageNetDataset(train_set, self.norm_layer)
        test_set = ImageFolder(self.test_path, transform=self.tr_test)
        test_set = ImageNetDataset(test_set, self.norm_layer)
        train_set.targets = np.array(train_set.targets)
        rng = np.random.RandomState(seed)
        valid_set = copy.deepcopy(train_set)
        valid_idx = []
        for i in range(max(train_set.targets) + 1):
            class_idx = np.where(train_set.targets == i)[0]
            valid_idx.append(
                rng.choice(class_idx, int(0.0 * len(class_idx)), replace=False)
            )
        valid_idx = np.hstack(valid_idx)
        train_set_copy = copy.deepcopy(train_set)

        valid_set.img_paths = train_set_copy.img_paths[valid_idx]
        valid_set.targets = train_set_copy.targets[valid_idx]

        train_idx = list(set(range(len(train_set))) - set(valid_idx))

        train_set.img_paths = train_set_copy.img_paths[train_idx]
        train_set.targets = train_set_copy.targets[train_idx]
        if class_to_replace is not None and indexes_to_replace is not None and num_classes_to_replace is not None:
            raise ValueError(
                "Only one of `class_to_replace` and `indexes_to_replace` and 'num_classes_to_replace 'can be specified"
            )
        if class_to_replace is not None and class_to_replace != -1:
            replace_class(
                train_set,
                class_to_replace,
                num_indexes_to_replace=num_indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )
            if num_indexes_to_replace is None:
                test_set.targets = np.array(test_set.targets)
                test_set.img_paths = test_set.img_paths[test_set.targets != class_to_replace]
                test_set.targets = test_set.targets[
                    test_set.targets != class_to_replace
                ]
                test_set.targets = test_set.targets.tolist()
        if indexes_to_replace is not None:
            replace_indexes(
                dataset=train_set,
                indexes=indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )
        if num_classes_to_replace is not None:
            class_idx = np.arange(num_classes_to_replace)
            replace_multi_class(
                dataset=train_set,
                multi_class_to_replace=class_idx,
                seed=seed - 1,
                only_mark=only_mark,
            )
        loader_args = {"num_workers": 0, "pin_memory": False}

        def _init_fn(worker_id):
            np.random.seed(int(seed))

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        val_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, val_loader, test_loader

def cifar10_dataloaders(
    batch_size=128,
    data_dir="datasets/cifar10",
    num_workers=2,
    random_to_replace: int = None,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
):
    # if no_aug:
    #     train_transform = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #         ]
    #     )
    # else:
    #     train_transform = transforms.Compose(
    #         [
    #             transforms.RandomCrop(32, padding=4),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #         ]
    #     )

    # test_transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #     ]
    # )
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    train_transform=transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    test_transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    print(
        "Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)

    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == 4500:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def replace_indexes(
    dataset: torch.utils.data.Dataset, indexes, seed=0, only_mark: bool = False
):
    if not only_mark:
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(
            list(set(range(len(dataset))) - set(indexes)), size=len(indexes)
        )
        dataset.data[indexes] = dataset.data[new_indexes]
        try:
            dataset.targets[indexes] = dataset.targets[new_indexes]
        except:
            dataset.labels[indexes] = dataset.labels[new_indexes]
        else:
            dataset._labels[indexes] = dataset._labels[new_indexes]
    else:
        # Notice the -1 to make class 0 work
        try:
            dataset.targets[indexes] = -dataset.targets[indexes] - 1
        except:
            try:
                dataset.labels[indexes] = -dataset.labels[indexes] - 1
            except:
                dataset._labels[indexes] = -dataset._labels[indexes] - 1


def replace_class(
    dataset: torch.utils.data.Dataset,
    class_to_replace: int,
    num_indexes_to_replace: int = None,
    seed: int = 0,
    only_mark: bool = False,
):
    if class_to_replace == -1:
        try:
            indexes = np.flatnonzero(np.ones_like(dataset.targets))
        except:
            try:
                indexes = np.flatnonzero(np.ones_like(dataset.labels))
            except:
                indexes = np.flatnonzero(np.ones_like(dataset._labels))
    else:
        try:
            indexes = np.flatnonzero(np.array(dataset.targets) == class_to_replace)
        except:
            try:
                indexes = np.flatnonzero(np.array(dataset.labels) == class_to_replace)
            except:
                indexes = np.flatnonzero(np.array(dataset._labels) == class_to_replace)

    if num_indexes_to_replace is not None:
        assert num_indexes_to_replace <= len(
            indexes
        ), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
        rng = np.random.RandomState(seed)
        indexes = rng.choice(indexes, size=num_indexes_to_replace, replace=False)
        print(f"Replacing indexes {indexes}")
    replace_indexes(dataset, indexes, seed, only_mark)

def replace_multi_class(
    dataset: torch.utils.data.Dataset,
    multi_class_to_replace: list = None,
    seed: int = 0,
    only_mark: bool = False,
):
    assert multi_class_to_replace is not None, "multi_class_to_replace should not be None"
    try:
        indexes = np.where(np.isin(np.array(dataset.targets), multi_class_to_replace))[0]
    except:
        try:
            indexes = np.where(np.isin(np.array(dataset.labels), multi_class_to_replace))[0]
        except:
            indexes = np.where(np.isin(np.array(dataset._labels), multi_class_to_replace))[0]
    print(f"Replacing indexes {indexes}")
    replace_indexes(dataset, indexes, seed, only_mark)

if __name__ == "__main__":
    # train_loader, val_loader, test_loader = cifar10_dataloaders()
    # for i, (img, label) in enumerate(train_loader):
    #     print(torch.unique(label).shape)
    import time
    time_start = time.time()
    mark_loader, _,_ = ImageNet({}).data_loaders(num_classes_to_replace=900,only_mark=True)
    marked = mark_loader.dataset.targets >= 0
    print(f"marked data size: {len(mark_loader.dataset.targets[marked])}")
    time_end = time.time()
    print(f"Time taken to load ImageNet dataset: {time_end - time_start} seconds")


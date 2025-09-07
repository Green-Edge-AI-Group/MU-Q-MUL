import argparse
import os
import pdb
import pickle
import random
import shutil
import time
from copy import deepcopy

import arg_parser
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from trainer import train, validate
from utils import *
from utils import NormalizeByChannelMeanStd

best_sa = 0


def main():
    global args, best_sa
    args = arg_parser.parse_args()

    torch.cuda.set_device(int(args.gpu))
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset
    if args.dataset == "imagenet":
        args.class_to_replace = None
        model, train_loader, val_loader = setup_model_dataset(args)
    else:
        (
            model,
            train_loader,
            val_loader,
            test_loader,
            marked_loader,
        ) = setup_model_dataset(args)
    model.cuda()

    print(f"number of train dataset {len(train_loader.dataset)}")
    print(f"number of val dataset {len(val_loader.dataset)}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.imagenet_arch:
        lambda0 = (
            lambda cur_iter: (cur_iter + 1) / args.warmup
            if cur_iter < args.warmup
            else (
                0.5
                * (
                    1.0
                    + np.cos(
                        np.pi * ((cur_iter - args.warmup) / (args.epochs - args.warmup))
                    )
                )
            )
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    else:
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=decreasing_lr, gamma=0.1
        # )  # 0.1 is fixed
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100, eta_min=1e-5)
    if args.resume:
        print("resume from checkpoint {}".format(args.checkpoint))
        checkpoint = torch.load(
            args.checkpoint, map_location=torch.device("cuda:" + str(args.gpu))
        )
        best_sa = checkpoint["best_sa"]
        start_epoch = checkpoint["epoch"]
        all_result = checkpoint["result"]

        model.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # initalization = checkpoint["init_weight"]
        print("loading from epoch: ", start_epoch, "best_sa=", best_sa)
        # print(all_result)


    print("Performance on the test data set")
    test_tacc = validate(test_loader, model, criterion, args)
    print("test acc: ",test_tacc)

if __name__ == "__main__":
    main()
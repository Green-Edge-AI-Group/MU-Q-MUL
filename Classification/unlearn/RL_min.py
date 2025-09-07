import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from torch.autograd import grad
import torch.nn.functional as F
from .impl import iterative_unlearn

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, y_hat, y, weights):
        
        losses = F.cross_entropy(y_hat, y, reduction='none')
        weighted_losses = weights * losses
        
        weighted_batch_loss = torch.sum(weighted_losses) / weighted_losses.numel()
        
        return weighted_batch_loss
    
def get_require_grad_params(model: torch.nn.Module, named=False):
    if named:
        return [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
    else:
        return [param for param in model.parameters() if param.requires_grad]


def sam_grad(model, loss, args):
    names = []
    params = []

    for param in get_require_grad_params(model, named=False):
        params.append(param)

    sample_grad = grad(loss, params, allow_unused=True)

    return sample_grad

def get_min_distance_labels(output,masks,target):

    confusable_target = target.clone()
    
   
    forget_output = output[masks == 1]
    forget_prob = F.softmax(forget_output, dim=-1)
    forget_target = target[masks == 1]
    
    gt_probabilities = torch.gather(forget_prob, 1, forget_target.unsqueeze(1)).squeeze()
    
    distance_probabilities = torch.abs(forget_output - gt_probabilities.unsqueeze(1))

    min_distance_indices = torch.zeros_like(target)
    
    for i in range(len(forget_target)):
        if masks[i] != 1: 
            continue
    
        gt_class = forget_target[i]
        
        mask = torch.ones_like(distance_probabilities[i], dtype=torch.bool)
        mask[gt_class] = False
        
        second_smallest_distance_index = torch.argmin(distance_probabilities[i][mask])
        
        min_distance_indices[i] = second_smallest_distance_index

    confusable_target[masks == 1] = min_distance_indices

    return confusable_target


def adjust_weights_based_on_gradients(model, forget_loader, retain_loader, args, criterion, optimizer):
    forget_grad_norms = []
    retain_grad_norms = []
    
    for i, (image, target) in enumerate(forget_loader):
        image = image.cuda()
        target = target.cuda()
        
        output_clean = model(image)
        loss = criterion(output_clean, target)
        
        sample_grad=sam_grad(model,loss,args)

        grad_norm=0.0
        with torch.no_grad():
            for param_grad in sample_grad:
                if param_grad is not None:
                    grad_norm+=param_grad.norm(2).item()
                else:
                    pass
        forget_grad_norms.append(grad_norm)
        model.zero_grad()

    
    for i, (image, target) in enumerate(retain_loader):
        image = image.cuda()
        target = target.cuda()
        
        output_clean = model(image)
        loss = criterion(output_clean, target)
        
        sample_grad=sam_grad(model,loss,args)
        
        grad_norm=0.0
        with torch.no_grad():
            for param_grad in sample_grad:
                if param_grad is not None:
                    grad_norm+=param_grad.norm(2).item()
                else:
                    pass
        retain_grad_norms.append(grad_norm)
        model.zero_grad()

    
    forget_avg_grad_norm = np.mean(forget_grad_norms)
    retain_avg_grad_norm = np.mean(retain_grad_norms)

    
    if forget_avg_grad_norm > retain_avg_grad_norm:
        retain_weight =  forget_avg_grad_norm / (forget_avg_grad_norm + retain_avg_grad_norm)
        forget_weight = retain_avg_grad_norm / (forget_avg_grad_norm + retain_avg_grad_norm)
        print("forget_avg_grad_norm > retain_avg_grad_norm")
    else:
        forget_weight = retain_avg_grad_norm / (forget_avg_grad_norm + retain_avg_grad_norm)
        retain_weight = forget_avg_grad_norm / (forget_avg_grad_norm + retain_avg_grad_norm)
        print("forget_avg_grad_norm < retain_avg_grad_norm")  
    return forget_weight, retain_weight


@iterative_unlearn
def RL_min(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    forget_dataset = deepcopy(forget_loader.dataset)    
    retain_dataset = retain_loader.dataset
    if not args.mrl:
    # if not False:
        if args.dataset == "cifar100" or args.dataset == "cifar10"or args.dataset == "eurosat":
            forget_dataset.targets = np.random.randint(0, args.num_classes, forget_dataset.targets.shape)
        elif args.dataset == "TinyImagenet":
            forget_dataset.dataset.targets = np.random.randint(0, args.num_classes, len(forget_dataset.dataset.targets))
        elif args.dataset == "svhn":
            forget_dataset.labels =  np.random.randint(0, args.num_classes, forget_dataset.labels.shape)
        else:
            raise ValueError("Dataset not supprot yet !")
    else:
        confusable_targets=[]
        with torch.no_grad():
            for image,target in forget_loader:
                image = image.cuda()
                target = target.cuda()
                output_clean = model(image)
                confusable_target = get_min_distance_labels(output_clean,torch.ones_like(target),target)
                confusable_targets.append(confusable_target.cpu())
            confusable_targets=torch.cat(confusable_targets,dim=0)   
        if args.dataset == "cifar100" or args.dataset == "cifar10"or args.dataset == "eurosat":
            forget_dataset.targets = confusable_targets.numpy().reshape(forget_dataset.targets.shape)
        elif args.dataset == "TinyImagenet":
            forget_dataset.targets = confusable_targets.numpy().reshape(forget_dataset.targets.shape)
        elif args.dataset == "svhn":
            forget_dataset.labels = confusable_targets.numpy().reshape(forget_dataset.labels.shape)
        else:
            raise ValueError("Dataset not supprot yet !")


    new_forget_loader=torch.utils.data.DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=True)
    if args.dataset == "cifar100" or args.dataset == "TinyImagenet"or args.dataset == "eurosat":
            
        forget_weight, retain_weight = adjust_weights_based_on_gradients(model, new_forget_loader, retain_loader, args, criterion, optimizer)
        

        try:
            forget_dataset.targets = - forget_dataset.targets - 1
        except:
            forget_dataset.dataset.targets = - forget_dataset.dataset.targets - 1


        train_dataset = torch.utils.data.ConcatDataset([forget_dataset, retain_dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        weighted_loss=WeightedCrossEntropyLoss()

        model.train()
      
        start = time.time()
        loader_len = len(new_forget_loader) + len(retain_loader)
      
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i+1, optimizer,
                            one_epoch_step=loader_len, args=args)
      
        for it, (image, target) in enumerate(train_loader):
            i = it
            image = image.cuda()
            target = target.cuda()
            # masks  binary matrix represent 1 forget 0 retain
            masks = torch.where(target < 0,1,0)
            target = torch.where(target < 0,-(target + 1),target)
            weights = torch.where(masks == 1,forget_weight,retain_weight)
            output_clean = model(image)

            loss = weighted_loss(output_clean,target,weights)
      
            optimizer.zero_grad()
            loss.backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
            optimizer.step()
      
            output = output_clean.float()
            loss = loss.float()
            prec1 = utils.accuracy(output.data, target)[0]
      
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
      
            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Reweight Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Time {3:.2f}'.format(
                          epoch, i, loader_len, end-start, loss=losses, top1=top1))
                start = time.time()
      
    elif args.dataset == "cifar10" or args.dataset == "svhn":
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
      
        model.train()
      
        start = time.time()
        loader_len = len(new_forget_loader) + len(retain_loader)
      
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i+1, optimizer,
                            one_epoch_step=loader_len, args=args)

        forget_weight, retain_weight = adjust_weights_based_on_gradients(model, new_forget_loader, retain_loader, args, criterion, optimizer)
      
        for i, (image, target) in enumerate(new_forget_loader):
            image = image.cuda()
            target = target.cuda()

            output_clean = model(image)

            loss = criterion(output_clean, target) * forget_weight
            
            optimizer.zero_grad()
            loss.backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
            optimizer.step()
            
        for i, (image, target) in enumerate(retain_loader):
            image = image.cuda()
            target = target.cuda()
            
            output_clean = model(image)
            loss = criterion(output_clean, target) * retain_weight
            
            optimizer.zero_grad()
            loss.backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
            optimizer.step()
            
            output = output_clean.float()
            loss = loss.float()
            prec1 = utils.accuracy(output.data, target)[0]
            
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
            
            if (i + 1) % args.print_freq == 0:
               end = time.time()
               print('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Time {3:.2f}'.format(
                         epoch, i, loader_len, end-start, loss=losses, top1=top1))
               start = time.time()

    return top1.avg
import torch
from torch.nn import functional as F
from .impl import iterative_unlearn
import copy
import time
import utils
import utils_loss as KD_loss



        
@iterative_unlearn
def BT(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    assert args.learning_teacher != None and args.unlearning_teacher != None
    learning_teacher = args.learning_teacher
    unlearning_teacher = args.unlearning_teacher
    forget_dataset = copy.deepcopy(forget_loader.dataset)
    retain_dataset = retain_loader.dataset
    try:
        forget_dataset.targets = - forget_dataset.targets - 1
    except:
        forget_dataset.dataset.targets = - forget_dataset.dataset.targets - 1


    train_dataset = torch.utils.data.ConcatDataset([forget_dataset,retain_dataset])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    criterion = KD_loss.DistributionLoss()

    if epoch < args.warmup:
        utils.warmup_lr(epoch, i+1, optimizer,
                        one_epoch_step=loader_len, args=args)
    
    # switch to train mode
    model.train()

    # teacher switch to eval mode
    learning_teacher.eval()
    unlearning_teacher.eval()
    learning_teacher.no_grad=True
    unlearning_teacher.no_grad=True


    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    start = time.time()
    loader_len = len(forget_loader) + len(retain_loader)
    for it, (image, target) in enumerate(train_loader):
        i = it + len(forget_loader)
        image = image.cuda()
        labels = torch.zeros_like(target)
        labels.masked_fill_(target < 0 , 1)
        labels = labels.cuda()
        target = torch.where(target < 0 , -(target + 1) , target)
        target = target.cuda()
        output_logits = model(image)
        output = output_logits.float()
        with torch.no_grad():
            f_teacher_out = learning_teacher(image).float()
            u_teacher_out = unlearning_teacher(image).float()

        labels = torch.unsqueeze(labels, dim = 1)
        # label 1 means forget sample
        # label 0 means retain sample
        overall_teacher_out = (1-labels) * f_teacher_out + labels * u_teacher_out
        loss=criterion(output,overall_teacher_out)

        optimizer.zero_grad()
        loss.backward()
        
        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
        
        optimizer.step()
    
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]
    
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))
    
        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                    'KD Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Time {3:.2f}'.format(
                        epoch, i, loader_len, end-start, loss=losses, top1=top1))
            start = time.time()


    return top1.avg
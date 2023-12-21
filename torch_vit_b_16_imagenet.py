import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import PIL
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import os
import shutil
import sys

# from soft_labels import get_Softlabels
# import losses
from torch.optim.lr_scheduler import ReduceLROnPlateau

vit_logfile = open("vit_b_16_imagenet_logfile_2.txt", "a")
sys.stdout = vit_logfile

print("*" * 120)
print("ViT Training Started...")
# In[ ]:
torch.manual_seed(0)
torch.cuda.empty_cache()

## Using GPU
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


import torch


net = models.vit_b_16(pretrained=False)


# In[ ]:


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, lrs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lrs * (0.1 ** (epoch // 10))
    if lr <= 1e-8:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# print(net)


# In[ ]:


data = "./data/imagenet"
traindir = os.path.join(data, "train")
valdir = os.path.join(data, "val")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image_size = 224

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            # transforms.RandAugment(num_ops=2, magnitude=9), #28
            transforms.ToTensor(),
            normalize,
        ]
    ),
)
from torch.utils.data import random_split
torch.manual_seed(0)
sub_size = 32768
rem = len(train_dataset) - sub_size

train_dataset, _ = random_split(train_dataset, [sub_size, rem]) 


# In[ ]:


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

val_transforms = transforms.Compose(
    [
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ]
)

val_dataset = datasets.ImageFolder(valdir, val_transforms)
# val_dataset.__len__ = lambda: 
sub_size = 16384
rem = len(val_dataset) - sub_size
val_dataset, _ = random_split(val_dataset, [sub_size, rem])
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)


# In[ ]:


def train(train_loader, net, criterion, optimizer, epoch):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )
    net.train()
    end = time.time()
    for k, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        outputs = F.log_softmax(outputs, dim=1)
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if k % 300 == 0:
            progress.display(k)
    # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    # return top1.avg, top5.avg


# In[ ]:


def validate(val_loader, net, criterion):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )
    net.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            loss = criterion(outputs, labels)
            outputs = F.log_softmax(outputs, dim=1)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 300 == 0:
                progress.display(i)
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )
    return top1.avg, top5.avg, losses.avg


# In[ ]:


# optimizer = RMSpropTF(net.parameters(), lr=0.016, momentum=0.9, weight_decay=1e-5)
# lr_scheduler, num_epochs = create_scheduler(args, optimizer)
# scheduler = StepLRScheduler(optimizer, decay_t = 2.4, decay_rate = 0.97)

criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, "min")


# In[ ]:


num_epochs = 100
best_acc1 = 0
best_acc5 = 0
net = net.cuda()
# net.load_state_dict(torch.load('model_best.pth.tar')['state_dict'])
# optimizer.load_state_dict(torch.load('model_best.pth.tar')['optimizer'])
# validate(val_loader, net, criterion)


for epoch in range(num_epochs):
    adjust_learning_rate(optimizer, epoch, 0.1)
    train(train_loader, net, criterion, optimizer, epoch)
    acc1, acc5, val_loss = validate(val_loader, net, criterion)
    scheduler.step(val_loss)
    is_best = acc1 > best_acc1
    if is_best:
        filename = "weights/chkpt_epoch_" + str(epoch) + ".pt"
        torch.save(net.state_dict(), filename)
    best_acc1 = max(acc1, best_acc1)
    best_acc5 = max(acc5, best_acc5)
    save_checkpoint(
        {
            "epoch": epoch + 1,
            "arch": "weights/mobilenet",
            "state_dict": net.state_dict(),
            "best_acc1": best_acc1,
            "optimizer": optimizer.state_dict(),
        },
        is_best,
    )

vit_logfile.close()

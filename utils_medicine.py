import os
import sys
import json
import pickle
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
from torch.autograd import Variable

# import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc





class FocalLoss(nn.Module): 
    def __init__(
        self, class_num=38, alpha=None, gamma=2, size_average=True
    ):

        super(FocalLoss, self).__init__()  
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.0)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def train_one_epoch(model, optimizer, data_loader, device, epoch, y_true, y_score):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = FocalLoss(gamma=2)
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    acc_num = 0
    total_num = 0
    total_loss = 0.0
    for step, data in enumerate(data_loader):
        print("step is :", step)
        images, labels = data 
        sample_num += images.shape[0]  
        """type(images),len(images) is <class 'torch.Tensor'> 4 """
        labels = labels.to(device)
        images = images.to(torch.float32).to(device)
        pred = model(images) 
        pred_classes = torch.max(pred, dim=1)[1] 

        print("######################pred_classes#######################", pred_classes)
        y_true += labels.cpu().tolist()  
        y_score += torch.max(nn.Sigmoid()(pred), dim=1)[0].cpu().tolist()
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()  
        loss = loss_function(pred, labels.to(device)) 
        loss.requires_grad_(True)  
        loss.backward()  
        acc_num += accu_num
        accu_loss += (
            loss.item()
        ) 
        data_loader.desc = "[train epoch {}] loss: {:.4f}, acc: {:.4f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num
        )
        print("data_loader.desc is ", data_loader.desc)
        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training，loss：", loss)
            sys.exit(1)
        total_loss += accu_loss
        optimizer.step() 
        optimizer.zero_grad()  
        total_num += sample_num
    print(
        "this is :total_num,total_loss,accu_num for this epoch:",
        total_num,
        total_loss,
        accu_num,
    )
    return (total_loss.item() / (step + 1)), (acc_num.item() / total_num) 


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, y_true, y_score):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device) 
    accu_loss = torch.zeros(1).to(device) 
    accu_f1 = torch.zeros(1).to(device) 
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    image = []
    label = []
    for step, data in enumerate(data_loader):  # 评价指标画图
        images, labels = data
        image.append(images)
        label.append(labels)
        sample_num += images.shape[0]
        pred = model(images.to(torch.float32).to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        y_true += labels.cpu().tolist()
        y_score += torch.max(nn.Sigmoid()(pred), dim=1)[0].cpu().tolist()
        accu_f1 += f1_score(labels.cpu(), pred_classes.cpu(), average="macro")
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.4f}, acc: {:.4f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num
        )

    fpr, tpr, thre = roc_curve(y_true, y_score)
    aucc = auc(fpr, tpr)
    aucc = 1 - aucc
    plt.cla()
    plt.plot(tpr, fpr, color="darkred", label="roc area:(%0.2f)" % aucc)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    dirr = "./image/epoch"
    dirr = dirr + str(epoch)
    dirr = dirr + ".png"
    if accu_num.item() / sample_num > 0.75:
        plt.savefig(dirr)

    return (
        accu_loss.item() / (step + 1),
        accu_num.item() / sample_num,
    ) 


def train_try(model, data_loader, device, epoch, y_true, y_score):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    acc_num = 0
    total_num = 0
    total_loss = 0.0
    for step, data in enumerate(data_loader):
        print("step is :", step)
        images, labels = data 
        sample_num += images.shape[0] 
        labels = labels.to(device)
        images = images.to(torch.float32).to(device)
        pred = model(images)  
        pred_classes = torch.max(pred, dim=1)[1] 
        y_true += labels.cpu().tolist()  
        y_score += torch.max(nn.Sigmoid()(pred), dim=1)[0].cpu().tolist()
        accu_num += torch.eq(pred_classes, labels.to(device)).sum() 
        loss = loss_function(pred, labels.to(device)) 
        loss.requires_grad_(True) 
        loss.backward()  
        acc_num += accu_num
        accu_loss += (
            loss.item()
        ) 
        data_loader.desc = "[train epoch {}] loss: {:.4f}, acc: {:.4f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num
        )
        print("data_loader.desc is ", data_loader.desc)
        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training，loss：", loss)
            sys.exit(1)
        total_loss += accu_loss
        total_num += sample_num
    return (
        (total_loss.item() / (step + 1)),
        (acc_num.item() / total_num),
    )

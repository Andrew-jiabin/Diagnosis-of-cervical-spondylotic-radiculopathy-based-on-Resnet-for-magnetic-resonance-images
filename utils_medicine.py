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

"""

"utils" 这个名称是一种常见的命名习惯，它表示 "utilities"（工具）的缩写，意思是实用工具或辅助函数。
在深度学习框架中，"utils" 是一个常见的模块或包名称，通常用于存放一些通用的辅助函数、工具函数或实用工具类，
这些函数和类在不同的模型或任务中可能会被多次使用。

"""


# val_path = random.sample(
#     images, k=int(len(images) * val_rate)
# )  ##按比例随机采样出验证样本集合的路径


class FocalLoss(nn.Module):  ##是一种特殊的损失函数，对于易于判断的函数内容， ##损失函数可以改
    def __init__(
        self, class_num=38, alpha=None, gamma=2, size_average=True
    ):  ##用于初始化 FocalLoss 类的参数
        ##这里的class_num为什么是38？？？

        super(FocalLoss, self).__init__()  ##调用父类 nn.Module 的构造函数，初始化 FocalLoss 类
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average  ##FocalLoss特有属性

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
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

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
    # print("sys.stdout is ", sys.stdout)
    # sys.stdout is  <_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>
    acc_num = 0
    total_num = 0
    total_loss = 0.0
    for step, data in enumerate(data_loader):
        # print("enumerate(data_loader) is :", enumerate(data_loader))
        # 'enumerate(data_loader) is : <enumerate object at 0x000001577C8C8C70>'
        print("step is :", step)
        # print("data is :", len(data), data[0], data[1], data[1])
        """
        data is : 2 torch.Size([4, 21, 224, 224]) torch.Size([4]) tensor([0, 1, 1, 1])
        data is given in the form of 'labels of four,four batchs'
        batchsize is 4 or less
        """

        images, labels = data  # 获取输入图像和对应标签
        sample_num += images.shape[0]  # 更新样本总数
        """type(images),len(images) is <class 'torch.Tensor'> 4 """
        # images = images.to(torch.float32).to("cuda:0").unsqueeze(1)
        labels = labels.to(device)
        images = images.to(torch.float32).to(device)
        pred = model(images)  # 将图像输入模型，得到预测输出 # 添加了.softmax(dim=1) 9.12
        # print(" pred's content is :", pred)
        """ 
        pred's content is : tensor([[ 1.8863, -2.3806],
        [ 2.6064, -3.3310],
        [ 1.5030, -1.8082],
        [ 1.1952, -1.3597],
        [ 1.9885, -2.3649]], grad_fn=<AddmmBackward0>)
        """
        pred_classes = torch.max(pred, dim=1)[1]  # 获取预测输出中概率最大的类别索引

        print("######################pred_classes#######################", pred_classes)
        y_true += labels.cpu().tolist()  # 将真实标签转换为CPU上的Python列表并拼接到y_true中
        y_score += torch.max(nn.Sigmoid()(pred), dim=1)[0].cpu().tolist()
        # 获取预测输出中概率最大值，并将其转换为CPU上的Python列表并拼接到y_score中
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()  # 统计预测正确的样本数量
        loss = loss_function(pred, labels.to(device))  # 计算损失
        loss.requires_grad_(True)  # 设置损失的梯度为True，用于反向传播时计算梯度
        loss.backward()  # 反向传播计算梯度
        acc_num += accu_num
        accu_loss += (
            loss.item()
        )  # 累计损失，将损失从计算图中分离，避免梯度累积        原来是 ： accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.4f}, acc: {:.4f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num
        )
        # 在进度条上显示当前的损失和准确率
        print("data_loader.desc is ", data_loader.desc)
        if not torch.isfinite(loss):  # 检查损失是否为有限值，如果不是，则可能出现问题，终止训练
            print("WARNING: non-finite loss, ending training，loss：", loss)
            sys.exit(1)
        total_loss += accu_loss
        optimizer.step()  # 根据计算得到的梯度更新模型参数
        optimizer.zero_grad()  # 清空优化器的梯度缓存，为下一次计算梯度做准备
        total_num += sample_num
    print(
        "this is :total_num,total_loss,accu_num for this epoch:",
        total_num,
        total_loss,
        accu_num,
    )
    return (total_loss.item() / (step + 1)), (acc_num.item() / total_num)  # 返回平均损失和准确率


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, y_true, y_score):
    """
        calling process:

        val_loss, val_acc = evaluate(
        model=model,
        data_loader=val_loader,
        device=device,
        epoch=epoch,
        y_true=y_true,
        y_score=y_score,
    )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

        val_dataset = MyDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        transform=data_transform["val"],
    )
    """

    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = FocalLoss(gamma=2)
    model.eval()
    # .eval()的意义是打开evaluate模式，不是一般意义的eval函数
    # 设置模型为评估模式，这会冻结所有 BatchNormalization 和 Dropout 层
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_f1 = torch.zeros(1).to(device)  # f1 score是模型准确性的一种度量
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    # print(data_loader.type)
    image = []
    label = []
    for step, data in enumerate(data_loader):  # 评价指标画图
        images, labels = data
        image.append(images)
        label.append(labels)
        sample_num += images.shape[0]
        pred = model(images.to(torch.float32).to(device))
        # pred = model(images.to(torch.float32).to(device).unsqueeze(1))
        pred_classes = torch.max(pred, dim=1)[1]
        y_true += labels.cpu().tolist()
        # y_true += labels.cpu().tolist()
        y_score += torch.max(nn.Sigmoid()(pred), dim=1)[0].cpu().tolist()
        # print("______________","\n",pred,"\n",labels,"\n","______________",)
        accu_f1 += f1_score(labels.cpu(), pred_classes.cpu(), average="macro")
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        """data_loader.desc = "[va-epo {}]  {:.4f}:{:.4f}:{:.4f}".format(epoch,
                                                                      accu_loss.item() / (step + 1),
                                                                      accu_num.item() / sample_num,
                                                                      accu_f1.item() / (step + 1))
                                                                      """
        data_loader.desc = "[valid epoch {}] loss: {:.4f}, acc: {:.4f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num
        )

    fpr, tpr, thre = roc_curve(y_true, y_score)
    ##计算auc的值，就是roc曲线下的面积

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
    dirr = dirr + str(epoch)  ##这里不会少一个杠？？？
    dirr = dirr + ".png"
    if accu_num.item() / sample_num > 0.75:
        plt.savefig(dirr)

    return (
        accu_loss.item() / (step + 1),
        accu_num.item() / sample_num,
    )  ##这里都是在求平均值


# def train_try(model, optimizer, data_loader, device, epoch, y_true, y_score):
def train_try(model, data_loader, device, epoch, y_true, y_score):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = FocalLoss(gamma=2)
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    # optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    # print("sys.stdout is ", sys.stdout)
    # sys.stdout is  <_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>
    acc_num = 0
    total_num = 0
    total_loss = 0.0
    for step, data in enumerate(data_loader):
        # print("enumerate(data_loader) is :", enumerate(data_loader))
        # 'enumerate(data_loader) is : <enumerate object at 0x000001577C8C8C70>'
        print("step is :", step)
        # print("data is :", len(data), data[0], data[1], data[1])

        """
        data is : 2 torch.Size([4, 21, 224, 224]) torch.Size([4]) tensor([0, 1, 1, 1])
        data is given in the form of 'labels of four,four batchs'
        batchsize is 4 or less
        """

        images, labels = data  # 获取输入图像和对应标签
        sample_num += images.shape[0]  # 更新样本总数
        """type(images),len(images) is <class 'torch.Tensor'> 4 """
        # images = images.to(torch.float32).to("cuda:0").unsqueeze(1)
        labels = labels.to(device)
        images = images.to(torch.float32).to(device)
        pred = model(images)  # 将图像输入模型，得到预测输出 # 添加了.softmax(dim=1) 9.12
        # print(" pred's content is :", pred)
        pred_classes = torch.max(pred, dim=1)[1]  # 获取预测输出中概率最大的类别索引
        y_true += labels.cpu().tolist()  # 将真实标签转换为CPU上的Python列表并拼接到y_true中
        y_score += torch.max(nn.Sigmoid()(pred), dim=1)[0].cpu().tolist()
        # 获取预测输出中概率最大值，并将其转换为CPU上的Python列表并拼接到y_score中
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()  # 统计预测正确的样本数量
        loss = loss_function(pred, labels.to(device))  # 计算损失
        loss.requires_grad_(True)  # 设置损失的梯度为True，用于反向传播时计算梯度
        loss.backward()  # 反向传播计算梯度
        acc_num += accu_num
        accu_loss += (
            loss.item()
        )  # 累计损失，将损失从计算图中分离，避免梯度累积        原来是 ： accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.4f}, acc: {:.4f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num
        )
        # 在进度条上显示当前的损失和准确率
        print("data_loader.desc is ", data_loader.desc)
        if not torch.isfinite(loss):  # 检查损失是否为有限值，如果不是，则可能出现问题，终止训练
            print("WARNING: non-finite loss, ending training，loss：", loss)
            sys.exit(1)
        total_loss += accu_loss
        total_num += sample_num
    return (
        (total_loss.item() / (step + 1)),
        (acc_num.item() / total_num),
    )  # 返回平均损失和准确率

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms, datasets
import torch.cuda
import torch


# 1:交叉熵
class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        # 获取预测值的数量和批次大小
        # num_samples = output.size(0)
        # 对预测值应用 softmax 函数，将其转换为概率分布
        predictions = torch.softmax(output, dim=1)
        # target = torch.nn.functional.one_hot(target, num_classes=10)
        # print('softmax之后:', predictions)
        # 使用预测值的概率分布和目标标签计算交叉熵损失

        # log_predictions = torch.log(predictions)
        # cross_entropy_loss = -torch.mean(target * log_predictions)

        log_predictions = torch.log(predictions)
        cross_entropy_loss = -torch.sum(target * log_predictions) / output.shape[0]
        return cross_entropy_loss


# 2:均方误差
class MseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, target):
        # expanded_target = target.unsqueeze(1).expand_as(predictions)
        # print(expanded_target)
        loss = torch.pow(predictions - target, 2).mean()
        # print(squared_error.size())
        return loss


# 3:绝对值误差
class AbsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, target):
        # 计算预测值与目标值之间的绝对值差
        # expanded_target = target.unsqueeze(1).expand_as(predictions)
        # print(expanded_target)
        absolute_diff = torch.abs(predictions - target)
        # 计算平均绝对值损失
        loss = torch.mean(absolute_diff)
        return loss


# 示例用法
# predictions = torch.rand(5, 10)  # 假设 batch_size 为 5，分类数量为 10
# targets = torch.randint(0, 10, (5,))  # 假设目标标签是随机生成的，范围在 0 到 9 之间
# print(predictions.shape, targets.shape)
# print(predictions.size(0))
# print(predictions.shape[0])
# cross_entropy_loss = CrossEntropy()
# mse_loss = MseLoss()
# abs_loss = AbsLoss()
# print('predictions：', predictions)
# print('target：', targets)
# after_softmax_predictions = torch.softmax(predictions, dim=1)
#
# print("交叉熵损失1:", cross_entropy_loss(predictions, targets).item())
# print("交叉熵损失2:", cross_entropy_loss(after_softmax_predictions, targets).item())
#
# loss_fjlk=nn.MSELoss()
# print("均方损失:", loss_fjlk(predictions,targets).item())
# print("均方损失:", mse_loss(predictions, targets).item())
# print("绝对值误差:", abs_loss(predictions, targets).item())

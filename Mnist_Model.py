import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms, datasets
import torch.cuda
import torch.nn.init as init
import torch.nn.functional as F


# 激活函数
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def softmax(x, dim):
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


# 单层感知机
class SinglePerception(nn.Module):
    def __init__(self, input_size, output_size, activate):
        super(SinglePerception, self).__init__()
        # 定义权重矩阵 W 和偏置向量 b
        self.W = nn.Parameter(torch.empty(output_size, input_size))  # 输出尺寸 x 输入尺寸
        self.b = nn.Parameter(torch.empty(output_size))  # 输出尺寸
        self.activate = activate
        # 初始化权重和偏置
        init.normal_(self.W)
        init.zeros_(self.b)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入展平成一维张量
        # print('dsa:', x.shape)
        # 计算 WX + b
        x = torch.matmul(x, torch.transpose(self.W, 0, 1)) + self.b
        if self.activate == 'sigmoid':
            x = sigmoid(x)
        if self.activate == 'softmax':
            x = softmax(x, dim=1)
        return x


# 多层感知机（3层）
class MultiPerception1(nn.Module):
    def __init__(self, in_size, out_size, activate):
        super().__init__()
        self.linear1 = nn.Linear(in_size, 512)
        self.linear2 = nn.Linear(512, 32)
        self.linear3 = nn.Linear(32, out_size)
        self.activate = activate

    def forward(self, data):
        linears = [self.linear1, self.linear2]
        data = data.reshape(-1, 28 * 28)
        if self.activate == 'Sigmoid':
            for linear in linears:
                data = F.sigmoid(linear(data))
            data = self.linear3(data)
            return data
        elif self.activate == 'Softmax':
            for linear in linears:
                data = F.softmax(linear(data), dim=1)
            data = self.linear3(data)
            return data
        elif self.activate == 'Relu':
            for linear in linears:
                data = F.relu(linear(data))
            data = self.linear3(data)
            return data
        elif self.activate == 'LeakyRelu':
            for linear in linears:
                data = F.leaky_relu(linear(data), negative_slope=0.5)
            data = self.linear3(data)
            return data
        elif self.activate == 'Tanh':
            for linear in linears:
                data = F.tanh(linear(data))
            data = self.linear3(data)
            return data
        elif self.activate == 'Elu':
            for linear in linears:
                data = F.elu(linear(data))
            data = self.linear3(data)
            return data
        elif self.activate == 'SoftPlus':
            for linear in linears:
                data = F.softplus(linear(data))
            data = self.linear3(data)
            return data


# 多层感知机（5层）
class MultiPerception2(nn.Module):
    def __init__(self, in_size, out_size, activate):
        super().__init__()
        self.linear1 = nn.Linear(in_size, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 32)
        self.linear4 = nn.Linear(32, 32)
        self.linear5 = nn.Linear(32, out_size)
        self.activate = activate

    def forward(self, data):
        # print(data.shape)
        linears = [self.linear1, self.linear2, self.linear3, self.linear4]
        data = data.reshape(-1, 28 * 28)
        if self.activate == 'Sigmoid':
            for linear in linears:
                data = F.sigmoid(linear(data))
            data = self.linear5(data)
            return data
        elif self.activate == 'Softmax':
            for linear in linears:
                data = F.softmax(linear(data), dim=1)
            data = self.linear5(data)
            return data
        elif self.activate == 'Relu':
            for linear in linears:
                data = F.relu(linear(data))
            data = self.linear5(data)
            return data
        elif self.activate == 'LeakyRelu':
            for linear in linears:
                data = F.leaky_relu(linear(data), negative_slope=0.5)
            data = self.linear5(data)
            return data
        elif self.activate == 'Tanh':
            for linear in linears:
                data = F.tanh(linear(data))
            data = self.linear5(data)
            return data
        elif self.activate == 'Elu':
            for linear in linears:
                data = F.elu(linear(data))
            data = self.linear5(data)
            return data
        elif self.activate == 'SoftPlus':
            for linear in linears:
                data = F.softplus(linear(data))
            data = self.linear5(data)
            return data


# 卷积层（2层）
class ConvLayer(nn.Module):
    def __init__(self, activate):
        super().__init__()
        self.activate = activate
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.linear1 = nn.Linear(24 * 24 * 32, 10)

    def forward(self, data):
        # print(data.shape)

        if self.activate == 'Sigmoid':
            data = F.sigmoid(self.conv1(data))
            data = F.sigmoid(self.conv2(data))
            data = data.view(data.size(0), -1)
            data = self.linear1(data)
            return data
        elif self.activate == 'Softmax':
            data = F.softmax(self.conv1(data),dim=1)
            data = F.softmax(self.conv2(data),dim=1)
            data = data.view(data.size(0), -1)
            data = self.linear1(data)
            return data
        elif self.activate == 'Relu':
            data = F.relu(self.conv1(data))
            data = F.relu(self.conv2(data))
            data = data.view(data.size(0), -1)
            data = self.linear1(data)
            return data
        elif self.activate == 'LeakyRelu':
            data = F.leaky_relu(self.conv1(data),negative_slope=0.5)
            data = F.leaky_relu(self.conv2(data),negative_slope=0.5)
            data = data.view(data.size(0), -1)
            data = self.linear1(data)
            return data
        elif self.activate == 'Tanh':
            data = F.tanh(self.conv1(data))
            data = F.tanh(self.conv2(data))
            data = data.view(data.size(0), -1)
            data = self.linear1(data)
            return data
        elif self.activate == 'Elu':
            data = F.elu(self.conv1(data))
            data = F.elu(self.conv2(data))
            data = data.view(data.size(0), -1)
            data = self.linear1(data)
            return data
        elif self.activate == 'SoftPlus':
            data = F.softplus(self.conv1(data))
            data = F.softplus(self.conv2(data))
            data = data.view(data.size(0), -1)
            data = self.linear1(data)
            return data


# net = ConvLayer('Relu')
# a = torch.rand([32,1,28,28])
# b = net(a)
# print(a)
# print(b)
# a = torch.rand(32, 1, 28, 28)
# print(a.reshape(-1, 28 * 28).shape)

# jdk = Perceptron(28*28, 10)
# print(jdk.W.shape)
# print(jdk.b.shape)
# print(jdk.W.T.shape)
# a = torch.rand(64,1,28,28,requires_grad=True)
# print(a.shape)

# print(jdk(a))
# print(jdk.softmax(x).shape)

# y_pred = torch.rand([5, 10])
# # print(y_pred)
# target = torch.tensor([1, 2, 0, 3, 2])
# print(target)
# y = torch.nn.functional.one_hot(target, num_classes=10)
# print(y)
#
# log_predictions = torch.log(y_pred)
# # print(log_predictions)
# # print(y.shape)
# cross_entropy_loss2 = -torch.sum(y * log_predictions) / y_pred.shape[0]
# print('cross_entropy_loss2:', cross_entropy_loss2)
# x=torch.tensor([5,6,2,3,5])
# expanded_target = x.unsqueeze(1)
# print(expanded_target)

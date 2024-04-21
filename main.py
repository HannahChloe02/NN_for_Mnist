import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms, datasets
import torch.cuda
from torchvision.datasets import MNIST
import Loss_Function
import Mnist_Model
from Config import parse_args
from Dataload import get_dataloader
from Train_Test import train_part, test_part
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# import matplotlib.pyplot as plt


# 展示部分数据
# def show_data(train_loader, test_loader):
#     exm = enumerate(train_loader)
#     batch_idx, (exm_data, exm_label) = next(exm)
#     fig = plt.figure()
#     for i in range(8):
#         plt.subplot(2, 4, i + 1)
#         plt.tight_layout()
#         plt.imshow(exm_data[i][0], cmap='gray', interpolation='none')
#         plt.title("Ground Truth: {}".format(exm_label[i]))
#         plt.xticks([])
#         plt.yticks([])
#     plt.show()


# 网络模型


# if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     train_dataloader, test_dataloader = get_dataloader("dataset")
#
#     # 定义网络
#     Net = Mnist_model(28 * 28)
#     Net.to(device)
#     # 损失函数
#     loss_fn = nn.CrossEntropyLoss().to(device)
#     # 学习率
#     lr = 0.01
#     # 优化器
#     optim = torch.optim.Adam(Net.parameters(), lr=lr)
#
#     EPOCH = 300
#     epoch = 0
#     for epoch in range(0, EPOCH):
#         train_part(train_dataloader, Net, loss_fn, optim, epoch, device)
#         test_part(test_dataloader, Net, loss_fn, device

# def main(args):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     train_dataloader, test_dataloader = get_dataloader(args.data_path, args.batch_size)
#
#     # Define the network
#     Net = args.model(28 * 28)
#     Net.to(device)
#     # Loss function
#     loss_fn = nn.CrossEntropyLoss().to(device)
#     # Learning rate
#     lr = args.learning_rate
#     # Optimizer
#     opti = args.optimizer
#     optim = opti(Net.parameters(), lr=lr)
#
#     EPOCH = args.epochs
#     for epoch in range(EPOCH):
#         train_part(train_dataloader, Net, loss_fn, optim, epoch, device)
#         test_part(test_dataloader, Net, loss_fn, device)


if __name__ == "__main__":
    args = parse_args()
    # print(args)
    # print('=======================================')
    # 获取运行的设备
    if args.device == "cuda":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # 激活函数
    activate_choice = {'1': 'Sigmoid', '2': 'Softmax', '3': 'Relu', '4': 'LeakyRelu',
                       '5': 'Tanh', '6': 'Elu', '7': 'SoftPlus'}
    # 模型
    Net = None
    Net_choice = {'1': 'SinglePerception', '2': 'MultiPerception1', '3': 'MultiPerception2', '4': 'ConvLayer'}
    if args.model == '1':
        Net = Mnist_Model.SinglePerception(28 * 28, 10, activate_choice[args.activation]).to(device)
    elif args.model == '2':
        Net = Mnist_Model.MultiPerception1(28 * 28, 10, activate_choice[args.activation]).to(device)
    elif args.model == '3':
        Net = Mnist_Model.MultiPerception2(28 * 28, 10, activate_choice[args.activation]).to(device)
    elif args.model == '4':
        Net = Mnist_Model.ConvLayer(activate_choice[args.activation]).to(device)
    # Net = Mnist_model(28 * 28)
    # 优化器
    lr = args.learning_rate
    optimizer_class = getattr(torch.optim, args.optimizer)
    optimizer = optimizer_class(Net.parameters(), lr=lr)
    # 损失函数
    loss_choice = {'1': 'CrossEntropy', '2': 'MseLoss', '3': 'AbsLoss'}
    loss_fn = None
    if args.loss == '1':
        loss_fn = Loss_Function.CrossEntropy()
    elif args.loss == '2':
        loss_fn = Loss_Function.MseLoss()
    elif args.loss == '3':
        loss_fn = Loss_Function.AbsLoss()

    train_dataloader, test_dataloader = get_dataloader(args.data_dir, args.batch_size)
    # 定义日志文件名
    log_filename = (f"logs/{Net_choice[args.model]}/{args.optimizer}/"
                    f"{loss_choice[args.loss]}/{activate_choice[args.activation]}/train_test_log.txt")
    # 模型保存文件夹
    save_model_path = (f"models/{Net_choice[args.model]}/{args.optimizer}/"
                       f"{loss_choice[args.loss]}/{activate_choice[args.activation]}")
    # 定义SummaryWriter文件夹
    SummaryWriter_dir = (f"SummaryWriter/{Net_choice[args.model]}/{args.optimizer}/"
                         f"{loss_choice[args.loss]}/{activate_choice[args.activation]}")
    writer = SummaryWriter(log_dir=SummaryWriter_dir)
    # print(save_model_path)

    # 打印参数
    print(f'device:{device}\tmodel:{Net_choice[args.model]}\toptimizer:{args.optimizer}'
          f'\tloss_function:{loss_choice[args.loss]}\tactivate_function:{activate_choice[args.activation]}')
    EPOCH = args.epochs
    for epoch in range(EPOCH):
        train_part(train_dataloader, Net, loss_fn, optimizer, epoch, device, writer, log_filename)
        test_part(test_dataloader, Net, loss_fn, epoch, device, writer, log_filename)
        torch.save(Net.state_dict(), f"{save_model_path}/Net{epoch}.pth")
        print('模型已保存')

    # print(args)
    # main(args)

    # exm = enumerate(train_dataloader)
    # batch_index,(exm_data,exm_label)=next(exm)
    # exm_data = exm_data.reshape(-1,28*28)
    # print("label:",exm_label.size(0))
    # print(exm_data.shape)
    # net = Mnist_model(exm_data.shape[1])
    # out = net(exm_data)
    # print("out的形状:",out.shape)
    # print(out)
    # print(batch_idx)
    # values,indices = torch.max(out,1)
    # print(values,indices)
    # print((indices==exm_label).sum().item())
    # print(len(train_dataloader.dataset))
    # print(len(train_dataloader))
    # show_data(train_dataloader,test_dataloader)
    # batch_count=0
    # print("exm",enumerate(exm))
    # for index, value in enumerate(exm):
    #     batch_count+=1
    # print(batch_count)
    # for batch_id,(image,label) in enumerate(train_dataloader):
    #     print(batch_id,image.shape,label.shape)

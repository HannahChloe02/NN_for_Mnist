import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch MNIST Example')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=150, help='Number of Epoch')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer to use')
    parser.add_argument('--model', type=str, choices=['1', '2', '3', '4'], required=True,
                        default='1', help='1 for SinglePerception, 2 for MLP1, 3 for MLP2, 4 for ConvLayer')
    parser.add_argument('--loss', type=str, choices=['1', '2', '3'], required=True,
                        default='1', help='Loss function to use, 1 for CrossEntropy, 2 for MseLoss, 3 for AbsLoss')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--activation', type=str, choices=['1', '2', '3', '4', '5', '6', '7'],
                        required=True, default='1', help='Activation function to use, 1 for Sigmoid, 2 for Softmax, '
                                                         '3 for Relu, 4 for LeakyRelu, 5 for Tanh, 6 for Elu,7 for '
                                                         'SoftPlus')
    args = parser.parse_args()
    return args

# print(parse_args())

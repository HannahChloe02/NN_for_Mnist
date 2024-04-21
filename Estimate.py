import torch


# 混淆矩阵
def confusion_matrix(y_true, y_pred, num_classes):
    # 初始化混淆矩阵
    cm = torch.zeros(num_classes, num_classes)

    # 遍历标签
    for t, p in zip(y_true, y_pred):
        # 增加对应位置的计数
        cm[t, p] += 1

    return cm


# 准确率
def accuracy(y_true, y_pred):
    return (y_true == y_pred).sum().item()


# 查全率
def recall(cm, num_classes):
    tp = torch.diag(cm)
    fn = torch.sum(cm, dim=1) - tp
    recall = tp / (tp + fn)
    return recall


# 查准率
def precision(cm, num_classes):
    tp = torch.diag(cm)
    fp = torch.sum(cm, dim=0) - tp
    precision = tp / (tp + fp)
    return precision


# F1分数
def f1_score(cm, num_classes):
    recall_value = recall(cm, num_classes)
    precision_value = precision(cm, num_classes)
    f1 = 2 * (precision_value * recall_value) / (precision_value + recall_value)
    return f1


# 示例用法
# num_classes = 3  # 假设有 3 个类别
# y_true = torch.tensor([0, 1, 2, 0, 1, 2])
# y_pred = torch.tensor([0, 2, 1, 0, 0, 1])
# cm = confusion_matrix(y_true, y_pred, num_classes)
# print("Confusion Matrix:")
# print(cm)
# print("Recall:", recall(cm, num_classes))
# print("Precision:", precision(cm, num_classes))
# print("F1 Score:", f1_score(cm, num_classes))

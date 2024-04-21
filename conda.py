import torch
import torch.cuda
import Estimate
def train_part(train_data, net, loss_fn, optimizer, epoch, device, writer, log_filename):
    net.train()
    total_train_loss = 0.0
    total_train_accuracy = 0.0
    train_data_size = len(train_data.dataset)
    y_true_list = []  # 存储真实标签
    y_pred_list = []  # 存储预测标签
    for batch_index, (image, label) in enumerate(train_data):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = net(image)
        label_new = torch.nn.functional.one_hot(label, num_classes=10).float()
        loss = loss_fn(output, label_new)
        total_train_loss += loss.item()
        values, indices = torch.max(output, 1)
        accuracy = Estimate.accuracy(indices, label)
        total_train_accuracy += accuracy
        # 将预测结果和真实标签存储到列表中
        y_true_list.extend(label.cpu().numpy())
        y_pred_list.extend(indices.cpu().numpy())
        loss.backward()
        optimizer.step()
    # 计算混淆矩阵
    cm = Estimate.confusion_matrix(torch.tensor(y_true_list), torch.tensor(y_pred_list), 10)
    # 计算评估指标
    recall_value = Estimate.recall(cm, 10)
    precision_value = Estimate.precision(cm, 10)
    f1_value = Estimate.f1_score(cm, 10)
    # 计算平均损失和准确率
    average_train_loss = total_train_loss / train_data_size
    average_train_accuracy = 100 * total_train_accuracy / train_data_size
    # 将评估指标写入 TensorBoard 日志并打印出来
    writer.add_scalar("train_loss", average_train_loss, epoch)
    writer.add_scalar("train_accuracy", average_train_accuracy, epoch)

    with open(log_filename, 'a') as log_file:
        log_entry = (f"Epoch:{epoch}\tTrain Average Loss:{average_train_loss}\t"
                     f"Train Average Accuracy:{average_train_accuracy}%\n"
                     f"TrainRecall:{recall_value}\nTrain Precision:{precision_value}\nTrain F1 Score:{f1_value}\n")
        log_file.write(log_entry)

    print(f"Epoch:{epoch}\tTrain Average Loss:{average_train_loss}\t"
          f"Train Average Accuracy:{average_train_accuracy}%")
    print(f"Train Recall:{recall_value}\nTrain Precision:{precision_value}\nTrain F1 Score:{f1_value}")


# 验证
def test_part(test_data, net, loss_fn, epoch, device, writer, log_filename):
    net.eval()
    total_test_loss = 0.0
    total_test_accuracy = 0.0
    average_test_loss = 0.0
    average_test_accuracy = 0.0
    test_data_size = len(test_data.dataset)
    y_true_list = []  # 存储真实标签
    y_pred_list = []  # 存储预测标签
    with torch.no_grad():
        for batch_index, (image, label) in enumerate(test_data):
            image, label = image.to(device), label.to(device)
            output = net(image)
            label_new = torch.nn.functional.one_hot(label, num_classes=10).float()
            loss = loss_fn(output, label_new)
            total_test_loss += loss
            values, indices = torch.max(output, 1)
            accuracy = (indices == label).sum().item()
            total_test_accuracy += accuracy
            # 将预测结果和真实标签存储到列表中
            y_true_list.extend(label.cpu().numpy())
            y_pred_list.extend(indices.cpu().numpy())
        # 计算混淆矩阵
        cm = Estimate.confusion_matrix(torch.tensor(y_true_list), torch.tensor(y_pred_list), 10)
        # 计算评估指标
        recall_value = Estimate.recall(cm, 10)
        precision_value = Estimate.precision(cm, 10)
        f1_value = Estimate.f1_score(cm, 10)
        # 计算平均损失和准确率
        average_test_loss = total_test_loss / test_data_size
        average_test_accuracy = 100 * total_test_accuracy / test_data_size
        # 将评估指标写入 TensorBoard 日志并打印出来
        writer.add_scalar("test_loss", average_test_loss, epoch)
        writer.add_scalar("test_accuracy", average_test_accuracy, epoch)
        with open(log_filename, 'a') as log_file:
            log_entry = (f"Epoch:{epoch}\tTest Average Loss:{average_test_loss}\t"
                         f"Test Average Accuracy:{average_test_accuracy}%\n"
                         f"Test Recall:{recall_value}\nTest Precision:{precision_value}\nTest F1 Score:{f1_value}\n")
            log_file.write(log_entry)

        print(f"Epoch:{epoch}\tTest Average Loss:{average_test_loss}\t"
              f"Test Average Accuracy:{average_test_accuracy}%")
        print(f"Test Recall:{recall_value}\nTest Precision:{precision_value}\nTest F1 Score:{f1_value}")

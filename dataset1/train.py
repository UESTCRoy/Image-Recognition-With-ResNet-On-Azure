import os
import sys
import json
import argparse
import mlflow

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
import time
from torch.optim.lr_scheduler import StepLR

from model import resnet34
from dataset import MyDataset


def main(batch_size, epochs):
    mlflow.set_experiment("ResNet-34")
    mlflow_run = mlflow.start_run()
    mlflow.autolog(log_models=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # all_text = ""
    # device_info = "using {} device.".format(device)
    # all_text += device_info + "\n"

    # batch_size = 64
    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    # worker_info = 'Using {} dataloader workers every process'.format(nw)
    # all_text += worker_info + "\n"

    train_label_file = 'train_label.txt'
    train_dataset = MyDataset(train_label_file, 'train')
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    validate_label_file = 'test_label.txt'
    validate_dataset = MyDataset(validate_label_file, 'test')
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4)

    print("using {} images for training, {} images for validation.".format(
        train_num, val_num))
    # image_info = "using {} images for training, {} images for validation.".format(train_num, val_num)
    # all_text += image_info + "\n"

    train_losses = []
    val_losses = []
    val_accuracies = []

    # net = torchvision.models.resnet34(pretrained=True)
    # num_features = net.fc.in_features
    # net.fc = nn.Linear(num_features, 10)  # 将最后的全连接层更改为您的类别数目
    net = resnet34()
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    # optimizer = optim.Adam(params, lr=0.0001)
    optimizer = optim.Adam(params, lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)  # Change the step_size and gamma as needed

    # epochs = 5
    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        # train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_loader):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        train_loss = running_loss / train_steps
        train_losses.append(train_loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_loss = 0.0
        val_loss_steps = 0
        with torch.no_grad():
            # val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_output = net(val_images.to(device))
                val_step_loss = loss_function(
                    val_output, val_labels.to(device))
                val_loss += val_step_loss.item()
                val_loss_steps += 1

                # val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num

        val_loss /= val_loss_steps
        val_losses.append(val_loss)
        val_accuracies.append(val_accurate)
        print('[epoch %d] train_loss: %.3f  val_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, train_loss, val_loss, val_accurate))
        
        # 每个epoch结束后更新学习率
        scheduler.step()

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')
    # mlflow.log_text(all_text, "notes.txt")

    # with mlflow.start_run():
    client = MlflowClient()
    run_id = mlflow.active_run().info.run_id
    timestamp = int(time.time() * 1000)
    # 记录train_loss指标
    client.log_batch(run_id,
                     metrics=[Metric(key="train_loss", value=val, timestamp=timestamp, step=i) for i, val in enumerate(train_losses)])

    # 记录val_loss指标
    client.log_batch(run_id,
                     metrics=[Metric(key="val_loss", value=val, timestamp=timestamp, step=i) for i, val in enumerate(val_losses)])

    # 记录val_accuracies指标
    client.log_batch(run_id,
                     metrics=[Metric(key="val_accuracies", value=val, timestamp=timestamp, step=i) for i, val in enumerate(val_accuracies)])

    # Plot training and validation loss
    fig_loss = plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    mlflow.log_figure(fig_loss, "loss_curve.png")

    # Plot validation accuracy
    fig_accuracy = plt.figure()
    plt.plot(val_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    mlflow.log_figure(fig_accuracy, "accuracy_curve.png")
    mlflow.end_run()


def predict_test(net, test_loader, output_file="test_label_predicted.txt"):
    net.eval()
    results = []
    with torch.no_grad():
        for test_data in test_loader:
            test_images, _ = test_data
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            results.extend(predict_y.cpu().numpy())

    with open(output_file, "w") as f:
        for i, label in enumerate(results):
            f.write(f"{i}.npy {label}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model with custom batch_size and epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train (default: 5)')
    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')
    main(args.batch_size, args.epochs)

# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import os
from utils import get_time_dif
from tensorboardX import SummaryWriter

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)


# 打印并记录到日志文件
def print_and_log(msg, log_file=None):
    print(msg)
    if log_file:
        log_file.write(str(msg) + '\n')
        log_file.flush()


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 创建日志文件
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    log_file_path = os.path.join(config.log_path, f'train_log_{time.strftime("%m-%d_%H.%M", time.localtime())}.txt')
    log_file = open(log_file_path, 'w', encoding='utf-8')

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    for epoch in range(config.num_epochs):
        print_and_log(f'Epoch [{epoch + 1}/{config.num_epochs}]', log_file)
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'.format(
                    total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve)
                print_and_log(msg, log_file)

                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print_and_log("No optimization for a long time, auto-stopping...", log_file)
                flag = True
                break
        if flag:
            break

    writer.close()
    log_file.close()
    test(config, model, test_iter, log_file_path)


def test(config, model, test_iter, log_file_path=None):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)

    lines = []
    lines.append('Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'.format(test_loss, test_acc))
    lines.append("Precision, Recall and F1-Score...")
    lines.append(test_report)
    lines.append("Confusion Matrix...")
    lines.append(str(test_confusion))
    lines.append("Time usage: " + str(get_time_dif(start_time)))

    # 打印和写入日志
    for line in lines:
        print(line)
    if log_file_path:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            for line in lines:
                f.write(str(line) + '\n')


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

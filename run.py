# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
import os
import pickle

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
# 默认按照char-level进行分词，具体见utils_fasttext.py中的build_dataset
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集 根地址
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'

    model_name = args.model
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)

    # 保证结果可复现
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    print("🔄 正在加载数据...")

    # 使用固定路径作为缓存位置
    vocab_path = os.path.join(dataset, 'data', 'vocab.pkl')
    data_cache_path = os.path.join(dataset, 'data', 'data_cache.pkl')
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

    if os.path.exists(vocab_path) and os.path.exists(data_cache_path):
        print(f"✅ 检测到缓存，加载 {vocab_path} 和 {data_cache_path}")
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        with open(data_cache_path, 'rb') as f:
            train_data, dev_data, test_data = pickle.load(f)
    else:
        print("📦 未检测到缓存，重新构建 vocab 和数据集")
        # args.word默认是char-level
        vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        with open(data_cache_path, 'wb') as f:
            pickle.dump((train_data, dev_data, test_data), f)

    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    print("✅ 数据加载完成")
    print("⏱️ 计时开始...")
    start_time = time.time()

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)

    time_dif = get_time_dif(start_time)
    print("⏳ 总用时:", time_dif)

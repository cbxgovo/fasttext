# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
# from utils_fasttext import build_dataset, build_iterator, get_time_dif
import argparse
import pickle as pkl
import pickle


class NewsClassifier:
    def __init__(self):
        print("NewsClassifier initializing!")
        self.parser = argparse.ArgumentParser(description='Chinese Text Classification')
        self.parser.add_argument('--model', default='FastText', type=str)
        self.parser.add_argument('--embedding', default='random', type=str)

        self.args = self.parser.parse_args()
        self.args.word = False
        self.dataset = 'THUCNews' # 修改 根地址 传到模型config 寻找保存的词表 模型文件等
        self.embedding = 'random'
        self.model_name = self.args.model
        self.labels = ["0", "1", "2","3","4","5"] # 修改 数据集预测标签 和训练的class.txt顺序对应

        x = import_module('models.' + self.model_name)
        self.config = x.Config(self.dataset, self.embedding)

        # 加速优化：取消 cudnn deterministic（若推理结果允许有微差）
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

        print("Loading data...")
        with open(self.config.vocab_path, 'rb') as f: # 加载已有的词表 如果有的话
            self.vocab = pickle.load(f)
        self.config.n_vocab = len(self.vocab)

        print("Loading model...")
        self.model = x.Model(self.config).to(self.config.device)
        if self.model_name != 'Transformer':
            init_network(self.model)
        self.model.load_state_dict(torch.load(self.config.save_path, map_location=self.config.device))
        self.model.eval()
        print("Model ready.")

    def my_to_tensor(self, config, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(config.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(config.device)
        bigram = torch.LongTensor([_[3] for _ in datas]).to(config.device)
        trigram = torch.LongTensor([_[4] for _ in datas]).to(config.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(config.device)
        return (x, seq_len, bigram, trigram)

    def str2numpy(self, text, config):
        UNK, PAD = '<UNK>', '<PAD>'
        tokenizer = lambda x: [y for y in x]  # char-level
        vocab = self.vocab

        def biGramHash(sequence, t, buckets):
            t1 = sequence[t - 1] if t - 1 >= 0 else 0
            return (t1 * 14918087) % buckets

        def triGramHash(sequence, t, buckets):
            t1 = sequence[t - 1] if t - 1 >= 0 else 0
            t2 = sequence[t - 2] if t - 2 >= 0 else 0
            return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

        def to_numpy(content, pad_size=32):
            token = tokenizer(content)
            seq_len = min(len(token), pad_size)
            token = token[:pad_size] + [PAD] * max(0, pad_size - len(token))
            words_line = [vocab.get(word, vocab.get(UNK)) for word in token]
            buckets = config.n_gram_vocab
            bigram = [biGramHash(words_line, i, buckets) for i in range(pad_size)]
            trigram = [triGramHash(words_line, i, buckets) for i in range(pad_size)]
            return [(words_line, -1, seq_len, bigram, trigram)]

        npy = to_numpy(text, config.pad_size)
        return self.my_to_tensor(config, npy)

    def classify(self, title):
        with torch.no_grad():
            data = self.str2numpy(title, self.config)
            outputs = self.model(data)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()[0]
            return self.labels[predic]


if __name__ == '__main__':
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    # input_file = 'THUCNews/data/0604.txt' # 修改 需要预测的数据集地址
    # output_file = 'THUCNews/data/0604_pre.txt'
    input_file = '/workspace/xumh3@xiaopeng.com/text_quality/output/train.txt' # 修改 需要预测的数据集地址 当前地址17389条测试数据集
    output_file = '/workspace/xumh3@xiaopeng.com/text_quality/output/train_predictions.txt'  # 修改 预测结果保存地址

    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        exit()

    classifier = NewsClassifier()

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    total_lines = len(lines)
    print(f"📄 待预测文本总数：{total_lines} 条")

    results = [None] * total_lines
    num_threads = min(8, os.cpu_count())

    print("🚀 开始多线程并行预测...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_index = {
            executor.submit(classifier.classify, line): idx
            for idx, line in enumerate(lines)
        }

        for future in tqdm(as_completed(future_to_index), total=total_lines, desc="预测中", ncols=80):
            idx = future_to_index[future]
            try:
                pred_label = future.result()
                results[idx] = (lines[idx], pred_label)
            except Exception as e:
                print(f"❌ 第 {idx} 行预测失败：{e}")
                results[idx] = (lines[idx], "ERROR")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for text, label in results:
            f_out.write(f"{text}\t{label}\n")

    end_time = time.time()
    print(f"\n✅ 所有预测完成，结果已保存到：{output_file}")
    print(f"⏱️ 总耗时：{end_time - start_time:.2f} 秒")

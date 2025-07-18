# coding: UTF-8
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# txt的每一行是一个待分类质量评分0-5的文本

class FastText(nn.Module):
    def __init__(self, embedding_pretrained, n_vocab, embed, dropout, hidden_size, num_classes, n_gram_vocab):
        super(FastText, self).__init__()
        if embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(n_vocab, embed, padding_idx=n_vocab - 1)
        self.embedding_ngram2 = nn.Embedding(n_gram_vocab, embed)
        self.embedding_ngram3 = nn.Embedding(n_gram_vocab, embed)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)
        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class NewsClassifier:
    def __init__(self):
        print("Initializing classifier...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 替换为你的绝对路径
        self.vocab_path = '/workspace-voice-ai/@*/llm/data/quality_data/fasttext/v0/pt_save_path/vocab.pkl'
        self.save_path = '/workspace-voice-ai/@*/llm/data/quality_data/fasttext/v0/pt_save_path/FastText.ckpt'

        with open(self.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        self.embedding_pretrained = None  # 如果有预训练embedding可以替换
        self.pad_size = 2048
        self.n_vocab = len(self.vocab)
        self.embed = 300
        self.dropout = 0.5
        self.hidden_size = 256
        self.num_classes = 6
        self.n_gram_vocab = 250499
        self.labels = ["0", "1", "2", "3", "4", "5"]

        self.model = FastText(
            self.embedding_pretrained, self.n_vocab, self.embed, self.dropout,
            self.hidden_size, self.num_classes, self.n_gram_vocab
        ).to(self.device)

        self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
        self.model.eval()

    def my_to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, bigram, trigram)

    def str2numpy(self, text):
        UNK, PAD = '<UNK>', '<PAD>'
        tokenizer = lambda x: [y for y in x]
        vocab = self.vocab

        def biGramHash(sequence, t, buckets):
            t1 = sequence[t - 1] if t - 1 >= 0 else 0
            return (t1 * 14918087) % buckets

        def triGramHash(sequence, t, buckets):
            t1 = sequence[t - 1] if t - 1 >= 0 else 0
            t2 = sequence[t - 2] if t - 2 >= 0 else 0
            return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

        def to_numpy(content):
            token = tokenizer(content)
            seq_len = min(len(token), self.pad_size)
            token = token[:self.pad_size] + [PAD] * max(0, self.pad_size - len(token))
            words_line = [vocab.get(word, vocab.get(UNK)) for word in token]
            bigram = [biGramHash(words_line, i, self.n_gram_vocab) for i in range(self.pad_size)]
            trigram = [triGramHash(words_line, i, self.n_gram_vocab) for i in range(self.pad_size)]
            return [(words_line, -1, seq_len, bigram, trigram)]

        return self.my_to_tensor(to_numpy(text))

    def classify(self, title):
        with torch.no_grad():
            data = self.str2numpy(title)
            outputs = self.model(data)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()[0]
            return self.labels[predic]


if __name__ == '__main__':
    input_file = '/workspace/xumh3@*.com/text_quality/output/test.txt'
    output_file = '/workspace/xumh3@*.com/text_quality/output/test_predictions.txt'

    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        exit()

    classifier = NewsClassifier()

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    total_lines = len(lines)
    print(f"📄 待预测文本总数：{total_lines} 条")

    results = [None] * total_lines
    # 它的作用是确定用于并发执行任务的线程数，取值为：
    # os.cpu_count()：获取当前机器的 CPU 核心数（逻辑核数，即线程数，而不是物理核）。
    # min(8, os.cpu_count())：取 8 和当前 CPU 核心数中的较小值，作为最大线程数。
    num_threads = min(8, os.cpu_count())
    # print(os.cpu_count()) #11
    # num_threads = os.cpu_count()  # 不做限制，完全使用所有核心

    print("🚀 开始预测...")
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

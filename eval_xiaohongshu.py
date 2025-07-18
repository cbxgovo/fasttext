# coding: UTF-8
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm


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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_path = '/workspace-voice-ai/@*/llm/data/quality_data/fasttext/v0/pt_save_path/vocab.pkl'
        self.save_path = '/workspace-voice-ai/@*/llm/data/quality_data/fasttext/v0/pt_save_path/FastText.ckpt'

        with open(self.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        self.embedding_pretrained = None
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

    def text_to_tensor(self, text):
        token_ids = [self.vocab.get(ch, self.vocab.get('<UNK>')) for ch in text]
        bigram_ids = [hash(tuple(text[i:i + 2])) % self.n_gram_vocab for i in range(len(text) - 1)]
        trigram_ids = [hash(tuple(text[i:i + 3])) % self.n_gram_vocab for i in range(len(text) - 2)]

        # padding
        def pad(seq, pad_size):
            if len(seq) < pad_size:
                seq += [self.vocab.get('<PAD>', self.n_vocab - 1)] * (pad_size - len(seq))
            else:
                seq = seq[:pad_size]
            return seq

        x = pad(token_ids, self.pad_size)
        bigrams = pad(bigram_ids, self.pad_size)
        trigrams = pad(trigram_ids, self.pad_size)

        x = torch.tensor([x], dtype=torch.long).to(self.device)
        bigrams = torch.tensor([bigrams], dtype=torch.long).to(self.device)
        trigrams = torch.tensor([trigrams], dtype=torch.long).to(self.device)
        return (x, None, bigrams, trigrams)

    def predict(self, text):
        with torch.no_grad():
            data = self.text_to_tensor(text)
            outputs = self.model(data)
            pred = torch.argmax(outputs, dim=1).item()
            return self.labels[pred]


def main():
    input_file = "testxiaohongshu.jsonl"
    output_file = "scored_testxiaohongshu.jsonl"

    classifier = NewsClassifier()

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, desc="Scoring"):
            try:
                item = json.loads(line)
                # jsonl格式 title content两个部分拼接作为文本内容输入，分类结果保存到新字段fasttext_score
                title = item.get("title", "")
                content = item.get("content", "")
                text = title + content
                label = classifier.predict(text)
                item["fasttext_score"] = label
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Error processing line: {e}")
                continue


if __name__ == "__main__":
    main()

from modelscope.msdatasets import MsDataset
import torch
from torch import nn
from config import *
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import itertools
import jieba  # 用于中文分词
from torch import optim
from utils import describe
from tqdm import tqdm

from model.transformer import Transformer

ds =  MsDataset.load('iic/WMT-Chinese-to-English-Machine-Translation-newstest', subset_name='default', split='test')

source_texts, target_texts = [], []

for data_item in ds:
    source_text = data_item['0']
    target_text = data_item['1']
    source_texts.append(source_text)
    target_texts.append(target_text)

print(f"source_texts size: {len(source_texts)}")
print(f"target_texts size: {len(target_texts)}")

# 1. 分词函数
def tokenize_zh(text):
    # 使用jieba进行中文分词
    return list(jieba.cut(text))

def tokenize_en(text):
    # 使用split进行英文分词
    return text.lower().split()

# 2. 构建词汇表
class Vocab:
    def __init__(self, tokens, min_freq=1, specials=["<unk>", "<pad>", "<bos>", "<eos>"]):
        # 统计词频
        counter = Counter(itertools.chain(*tokens))
        self.word2idx = {word: idx for idx, word in enumerate(specials)}
        
        for word, _ in counter.items():
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.unk_idx = self.word2idx["<unk>"]
        self.pad_idx = self.word2idx["<pad>"]
        self.bos_idx = self.word2idx["<bos>"]
        self.eos_idx = self.word2idx["<eos>"]

    def __len__(self):
        return len(self.word2idx)

    def word_to_index(self, word):
        return self.word2idx.get(word, self.unk_idx)

    def index_to_word(self, idx):
        return self.idx2word.get(idx, "<unk>")

    def encode(self, sentence, is_zh=True):
        if is_zh:
            tokens = tokenize_zh(sentence)
        else:
            tokens = tokenize_en(sentence)
        
        if is_zh:
            return [self.word_to_index(word) for word in tokens]
        return [self.bos_idx] + [self.word_to_index(word) for word in tokens]

    def decode(self, indices):
        return " ".join([self.index_to_word(idx) for idx in indices if idx not in [self.bos_idx, self.eos_idx, self.pad_idx]])

# 对源语言（中文）和目标语言（英文）进行分词
source_tokens = [tokenize_zh(text) for text in source_texts]
target_tokens = [tokenize_en(text) for text in target_texts]

# 创建源语言（中文）和目标语言（英文）的词汇表
source_vocab = Vocab(source_tokens)
target_vocab = Vocab(target_tokens)
print(f"source_vocab size: {len(source_vocab)}")
print(f"target_vocab size: {len(target_vocab)}")

# 3. 创建Dataset类
class TranslationDataset(Dataset):
    def __init__(self, source_texts, target_texts, source_vocab, target_vocab):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        # 将源句子（中文）和目标句子（英文）转换为索引
        source_encoded = self.source_vocab.encode(self.source_texts[idx], is_zh=True)
        target_encoded = self.target_vocab.encode(self.target_texts[idx], is_zh=False)
        return torch.tensor(source_encoded), torch.tensor(target_encoded)

# 4. 构建Dataloader
def collate_fn(batch):
    # 找出该批次中最长的序列长度
    source_batch, target_batch = zip(*batch)

    # 自定义填充/截断函数
    def pad_to_fixed_length(sequences, fixed_length, pad_value):
        padded_sequences = []
        for seq in sequences:
            if len(seq) < fixed_length:
                # 如果序列长度不足，则填充
                padded_seq = torch.cat([seq, torch.full((fixed_length - len(seq),), pad_value)])
            else:
                # 如果序列长度超出，则截断
                padded_seq = seq[:fixed_length]
            padded_sequences.append(padded_seq)
        return torch.stack(padded_sequences)

    # 进行填充，使得每个batch中的句子长度一致
    source_padded = pad_to_fixed_length(source_batch, max_len, source_vocab.pad_idx)
    target_padded = pad_to_fixed_length(target_batch, max_len, target_vocab.pad_idx)

    dec_outputs = target_padded[:, 1:]
    eos_tensor = torch.full((target_padded.size(0), 1), target_vocab.eos_idx)
    dec_outputs = torch.cat((dec_outputs, eos_tensor), dim=1)
    return source_padded, target_padded, dec_outputs



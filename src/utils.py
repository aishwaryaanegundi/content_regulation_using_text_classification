import os
from collections import Counter
import jieba
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

PAD = 0
UNK = 1

def make_vocab(train_file, result_dir="results", text_col_name=None):
    dic_filepath = os.path.join(result_dir,"vocab.txt")
    df = pd.read_csv(train_file, sep='\t')
    vocab2num = Counter()
    for sentence in df[text_col_name]:
        vocabs = jieba.lcut(sentence.strip())
        for vocab in vocabs:
            vocab = vocab.strip()
            if vocab and vocab != "":
                vocab2num[vocab] += 1
    with open(dic_filepath, "w", encoding="utf-8") as fw:
        fw.write("{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>"))
        for vocab, num in vocab2num.most_common():
            fw.write("{}\t{}\n".format(vocab, num))


def get_vocab(result_dir="results", min_count=1):
    vocabs = []
    with open(os.path.join(result_dir, "vocab.txt"), "r", encoding="utf-8") as fr:  
        for line in fr.readlines():
            lines = line.split()
            if int(lines[1]) >= min_count:
                vocabs.append(lines[0])
    vocab2idx = {vocab: idx for idx, vocab in enumerate(vocabs)}
    return vocab2idx

def load_data(file, max_len=100, vocab2idx=None, text_col_name=None, label_col_name=None, class_names=None):
   
    df = pd.read_csv(file, sep='\t')

    x_list = []
    for sentence in df[text_col_name].values:
        x = [vocab2idx.get(vocab, UNK) for vocab in jieba.cut(sentence)]
        x = x[:max_len]
        n_pad = max_len - len(x)
        x = x + n_pad * [PAD] 
        x_list.append(x)
    X = np.array(x_list, dtype=np.int64)
    
    if label_col_name:
        label2idx = {label: idx for idx, label in enumerate(class_names)}
        y = [label2idx[label] for label in df[label_col_name].values]
        y = np.array(y, dtype=np.int64)
    else:
        y = None

    return X, y

def get_labels(row, label_col_name):
    label = row[label_col_name]
    if (label == 0):
        return 'OK'
    elif (label == 1):
        return 'B'
    else:
        return None

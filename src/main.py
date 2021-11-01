import os
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from utils import make_vocab, get_vocab, load_data, get_labels
from models.cnn import CNNTextModel
from torch.utils.data.sampler import SubsetRandomSampler
import argparse

#Configurations
TRAIN_FILE = './data/KDC_train.tsv'
TEST_FILE = './data/KDC_test.tsv'
RESULT_DIR = './data/results/'
MODEL_DIR = './models/'
TEXT_COL_NAME = 'text'
LABEL_COL_NAME = 'status'
CLASS_NAMES = ['OK', 'B']
MIN_COUNT = 1
MAX_LEN = 100
BATCH_SIZE = 64
EMBEDDING_DIM = 128
EPOCHS = 10
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 32

MAKE_VOCAB = True
TRAIN = True
PREDICT = True


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    vocab2idx = get_vocab(result_dir=RESULT_DIR, min_count=MIN_COUNT)
    X_train, y_train = load_data(file=TRAIN_FILE, max_len=MAX_LEN, vocab2idx=vocab2idx, text_col_name=TEXT_COL_NAME,
                                 label_col_name=LABEL_COL_NAME, class_names=CLASS_NAMES)
    dataset_size = len(X_train)
    indices = list(range(dataset_size))
    split = int(np.floor(VALIDATION_SPLIT * dataset_size))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler = train_sampler)
    test_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE, sampler = valid_sampler)

    vocab_size = len(vocab2idx)

    # Build model.
    model = CNNTextModel(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, num_classes=len(CLASS_NAMES))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    data_size = len(train_dataset)
    batch_num = data_size // BATCH_SIZE + 1

    print("Training model..")
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        # Train model.
        model.train()
        batch = 1
        for batch_xs, batch_ys in train_loader:
            batch_xs = batch_xs.to(device)  
            batch_ys = batch_ys.to(device) 
            batch_out = model(batch_xs) 
            loss = criterion(batch_out, batch_ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch += 1
        checkpoint_path = os.path.join(MODEL_DIR, "model_epoch_{}.ckpt".format(epoch))
        torch.save(model, checkpoint_path)

        # Test model.
        model.eval()
        y_pred = []
        y_test = []
        for batch_xs, batch_ys in test_loader:
            batch_xs = batch_xs.to(device) 
            batch_out = model(batch_xs) 
            batch_pred = batch_out.argmax(dim=-1) 
            for i in batch_ys.cpu().numpy():
                y_pred.append(i)
            for i in batch_pred.cpu().numpy():
                y_test.append(i)
        y_pred = np.array(y_pred, dtype=np.int64)
        y_test = np.array(y_test, dtype=np.int64)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        f1_socre = metrics.f1_score(y_test, y_pred, average="macro")
        print("epoch {}, test accuracy {}, f1-score {}".format(epoch, accuracy, f1_socre))


def predict(epoch_idx):
    test_data = pd.read_csv(TEST_FILE, sep='\t')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(os.path.join(MODEL_DIR, "model_epoch_{}.ckpt".format(epoch_idx)))
    model = model.to(device)

    vocab2idx = get_vocab(result_dir=RESULT_DIR, min_count=MIN_COUNT)
    X, _ = load_data(TEST_FILE,
                     max_len=MAX_LEN,
                     vocab2idx=vocab2idx,
                     text_col_name=TEXT_COL_NAME)
    X = torch.from_numpy(X) 
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    y_pred = []
    for (batch_xs,) in loader:
        batch_xs = batch_xs.to(device) 
        batch_out = model(batch_xs) 
        batch_pred = batch_out.argmax(dim=-1) 
        for i in batch_pred.cpu().numpy():
            y_pred.append(i)
            
    test_data['status'] = y_pred
    test_data['status'] = test_data.apply(get_labels, label_col_name=LABEL_COL_NAME, axis = 1)
    test_data.to_csv(RESULT_DIR+'KDC test.tsv', sep='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-vocab", action="store_true",
                        help="Builds the vocab from training data.")
    parser.add_argument("--train", action="store_true",
                        help="Trains the model")
    parser.add_argument("--predict", action="store_true",
                        help="Predicts the labels for the test data.")

    args = parser.parse_args()

    if args.make_vocab:
        make_vocab(train_file=TRAIN_FILE, result_dir=RESULT_DIR, text_col_name=TEXT_COL_NAME)
    if args.train:
        train()
    if args.predict:
        predict(EPOCHS)
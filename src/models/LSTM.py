import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMTextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LSTMTextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,2,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x, _ = x
        out = self.embedding(x) 
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :]) 
        return out
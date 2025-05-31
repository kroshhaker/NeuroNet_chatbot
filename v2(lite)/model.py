import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1000, d_model))  # 1000 — максимум длины
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.token_emb(x) + self.pos_emb[:, :seq_len, :]
        x = self.transformer(x)
        return self.fc_out(x)


class TextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    # Добавляем метод инициализации скрытого состояния
    def init_hidden(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return (
            torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        )

    def forward(self, x, hidden):
        x = self.encoder(x)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        return out, (ht1, ct1)
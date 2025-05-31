import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import numpy as np
from model import TransformerModel

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/data2.txt", "r", encoding="utf-8") as f:
        text = f.read()

    with open("data/vocab.pkl", "rb") as f:
        vocab_data = pickle.load(f)

    stoi_raw = vocab_data['stoi']
    itos_raw = vocab_data['itos']

    max_index = max(itos_raw.values())
    itos = [None] * (max_index + 1)

    for c, i in itos_raw.items():
        itos[i] = c

    stoi = {}
    for c, i in stoi_raw.items():
        stoi[c] = i

    assert all(itos[stoi[c]] == c for c in stoi)

    vocab_size = max(itos_raw.values()) + 1

    def encode(s):
        return [stoi.get(c, stoi["<UNK>"]) for c in s]

    def decode(l):
        return ''.join([itos[i] for i in l])

    data = encode(text)
    data = np.array(data)

    # Гиперпараметры
    block_size = 256
    batch_size = 64
    lr = 3e-4
    n_iter = 1000

    def get_batch(data, block_size, batch_size):
        starts = np.random.randint(0, len(data) - block_size, (batch_size,))
        x = np.zeros((batch_size, block_size), dtype=np.int64)
        y = np.zeros((batch_size, block_size), dtype=np.int64)

        for i, start in enumerate(starts):
            x[i] = data[start:start + block_size]
            y[i] = data[start + 1:start + block_size + 1]

        return torch.from_numpy(x), torch.from_numpy(y)

    model = TransformerModel(vocab_size).to(device)

    if os.path.exists("data/model.pt"):
        model.load_state_dict(torch.load("data/model.pt", map_location=device))
        print("Загружены веса модели для дообучения")
    else:
        print("Обучение с нуля")

    weights = torch.ones(vocab_size).to(device)
    special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<SEP>', '\n', '\t', '\r']
    for token in special_tokens:
        if token in stoi:
            weights[stoi[token]] = 2.0

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    loss_history = []
    for iteration in range(n_iter):
        model.train()
        x, y = get_batch(data, block_size, batch_size)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (iteration + 1) % 50 == 0:
            avg_loss = np.mean(loss_history)
            print(f"Iteration {iteration + 1}/{n_iter}, Loss: {avg_loss:.4f}")

            # sample = generate_text(model, "Привет", length=200)
            # print(f"Генерация:\n{sample}\n{'-' * 50}")

            scheduler.step(avg_loss)
            loss_history = []

    # Сохранение модели
    torch.save(model.state_dict(), "data/model.pt")
    print("Обучение завершено, модель сохранена")
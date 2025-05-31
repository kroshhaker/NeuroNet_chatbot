import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import numpy as np
from model import TextRNN


def train_model(PATH_MODEL="../data/model_rnn.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Загрузка данных
    with open("../data/dialogues.csv", "r", encoding="utf-8") as f:
        text = f.read()

    # Загрузка словаря
    with open("../data/vocab.pkl", "rb") as f:
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
    vocab_size = max_index + 1  # Исправлено: vocab_size = max_index + 1

    def encode(s):
        return [stoi.get(c, stoi["<UNK>"]) for c in s]

    def decode(l):
        return ''.join([itos[i] for i in l if i < len(itos)])

    data = encode(text)
    data = np.array(data)

    # Гиперпараметры
    SEQ_LEN = 256  # Длина последовательности
    BATCH_SIZE = 64
    LR = 1e-3
    N_EPOCHS = 20000
    PRINT_EVERY = 50

    # Функция для создания батчей
    def get_batch():
        # Создаем единый массив вместо списка массивов
        inputs = np.zeros((BATCH_SIZE, SEQ_LEN), dtype=np.int64)
        targets = np.zeros((BATCH_SIZE, SEQ_LEN), dtype=np.int64)

        for i in range(BATCH_SIZE):
            start_idx = np.random.randint(0, len(data) - SEQ_LEN - 1)
            inputs[i] = data[start_idx:start_idx + SEQ_LEN]
            targets[i] = data[start_idx + 1:start_idx + SEQ_LEN + 1]

        return (
            torch.tensor(inputs, dtype=torch.long, device=device),
            torch.tensor(targets, dtype=torch.long, device=device)
        )

    # Инициализация модели
    model = TextRNN(input_size=vocab_size, hidden_size=256, embedding_size=128, n_layers=2).to(device)

    # Загрузка существующей модели
    if os.path.exists(PATH_MODEL):
        model.load_state_dict(torch.load(PATH_MODEL, map_location=device))
        print("Загружены веса модели для дообучения")
    else:
        print("Обучение с нуля")

    # Веса для классов (для особых токенов)
    weights = torch.ones(vocab_size).to(device)
    special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<SEP>', '\n', '\t', '\r']
    for token in special_tokens:
        if token in stoi:
            weights[stoi[token]] = 2.0

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)

    # Обучение
    print("Начинаем обучение...")
    loss_history = []

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        inputs, targets = get_batch()
        hidden = model.init_hidden(BATCH_SIZE)

        # Forward pass
        outputs, _ = model(inputs, hidden)

        # Reshape для вычисления потерь
        loss = criterion(
            outputs.view(-1, vocab_size),
            targets.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Обрезка градиентов
        optimizer.step()

        # Сохранение и вывод статистики
        loss_history.append(loss.item())

        if epoch % PRINT_EVERY == 0:
            avg_loss = np.mean(loss_history)
            print(f"Эпоха {epoch}/{N_EPOCHS}, Loss: {avg_loss:.4f}")
            loss_history = []

        if epoch % 500 == 0:
            torch.save(model.state_dict(), f"../data/model_rnn_{epoch /500}.pt")

    # Финальное сохранение модели
    torch.save(model.state_dict(), "../data/model_rnn.pt")
    print("Обучение завершено, модель сохранена")
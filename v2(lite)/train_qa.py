import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import numpy as np
import json
from torch.nn.utils.rnn import pad_sequence
from model import TextRNN


def train_model(PATH_MODEL="../data/model_rnn.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Загрузка данных
    with open("../data/dataset_mini.json", "r", encoding="utf-8") as f:
        examples = json.load(f)

    with open("../data/vocab.pkl", "rb") as f:
        vocab_data = pickle.load(f)

    stoi_raw = vocab_data['stoi']
    itos_raw = vocab_data['itos']

    max_index = max(itos_raw.values())
    itos = [None] * (max_index + 1)
    for c, i in itos_raw.items():
        itos[i] = c
    stoi = {c: i for c, i in stoi_raw.items()}
    vocab_size = max_index + 1

    def encode(s):
        return [stoi.get(c, stoi["<UNK>"]) for c in s]

    class Seq2SeqDataset(torch.utils.data.Dataset):
        def __init__(self, examples, max_inp_len=512, max_out_len=128):
            self.samples = []
            for item in examples:
                inp = item["input"].strip()
                out = item["output"].strip()
                inp_ids = encode(inp)[:max_inp_len]
                out_ids = encode(out + "<EOS>")[:max_out_len]
                if len(inp_ids) == 0 or len(out_ids) == 0:
                    continue
                self.samples.append((inp_ids, out_ids))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    dataset = Seq2SeqDataset(examples)
    print(f"Количество примеров: {len(dataset)}")

    def collate_fn(batch):
        inputs, targets = zip(*batch)
        inputs = [torch.tensor(x, dtype=torch.long) for x in inputs]
        targets = [torch.tensor(x, dtype=torch.long) for x in targets]
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=stoi['<PAD>'])
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=stoi['<PAD>'])
        return inputs_padded, targets_padded

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = TextRNN(input_size=vocab_size, hidden_size=256, embedding_size=128, n_layers=2).to(device)

    weights = torch.ones(vocab_size).to(device)
    special_tokens = ['<BOS>', '<EOS>', '<UNK>', '<SEP>', '\n', '\t', '\r']
    for token in special_tokens:
        if token in stoi:
            weights[stoi[token]] = 2.0

    criterion = nn.CrossEntropyLoss(ignore_index=stoi['<PAD>'], weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, amsgrad=True)

    if os.path.exists(PATH_MODEL):
        model.load_state_dict(torch.load(PATH_MODEL, map_location=device))
        print("Загружены веса модели для дообучения")
    else:
        print("Обучение с нуля")

    N_EPOCHS = 1000
    PRINT_EVERY = 1

    print("Начинаем обучение...")
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for inp_batch, out_batch in loader:
            inp_batch = inp_batch.to(device)
            out_batch = out_batch.to(device)

            optimizer.zero_grad()
            hidden = model.init_hidden(inp_batch.size(0), device)
            outputs, _ = model(inp_batch, hidden)

            min_len = min(outputs.size(1), out_batch.size(1))
            logits = outputs[:, :min_len, :].contiguous().view(-1, vocab_size)
            targets = out_batch[:, :min_len].contiguous().view(-1)

            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if epoch % PRINT_EVERY == 0:
            print(f"Эпоха {epoch}/{N_EPOCHS} - Loss: {avg_loss:.4f}")

        if epoch % 100 == 0:
            torch.save(model.state_dict(), f"../data/model_rnn_{epoch}.pt")

    torch.save(model.state_dict(), PATH_MODEL)
    print("Обучение завершено, модель сохранена.")


if __name__ == "__main__":
    train_model()

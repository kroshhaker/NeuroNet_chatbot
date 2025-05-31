# Не актуально

import torch
import torch.nn as nn
import pickle
import os
from torch.utils.data import DataLoader, Dataset
from model import TransformerModel

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка словаря
    with open("data/vocab.pkl", "rb") as f:
        data = pickle.load(f)

    stoi_raw = data['stoi']
    itos_raw = data['itos']

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

    import json
    with open("data/dataset_mini.json", "r", encoding="utf-8") as f:
        examples = json.load(f)

    class Seq2SeqDataset(Dataset):
        def __init__(self, examples, max_inp_len=2048, max_out_len=2048):
            self.data = []
            for item in examples:
                inp = item['input']
                out = item['output']
                inp_ids = encode(inp)[:max_inp_len]
                out_ids = encode(out + '<EOS>')[:max_out_len]
                if len(inp_ids) == 0 or len(out_ids) == 0:
                    continue
                self.data.append((inp_ids, out_ids))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            return self.data[i]

    ds = Seq2SeqDataset(examples)
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    model = TransformerModel(vocab_size).to(device)
    if os.path.exists("data/model.pt"):
        model.load_state_dict(torch.load("data/model.pt", map_location=device))
        print("=> загружены старые веса, продолжаем обучение")
    else:
        print("=> обучение с нуля")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 100
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0

        for inp_ids, out_ids in loader:
            inp = torch.tensor(inp_ids[0], dtype=torch.long, device=device).unsqueeze(0)  # [1, seq_in]
            tgt = out_ids[0]  # список int

            optimizer.zero_grad()
            loss = 0.0

            context = inp.clone()
            for t, target_token in enumerate(tgt):
                logits = model(context)
                last_logits = logits[0, -1, :].unsqueeze(0)  # [1, vocab_size]
                target = torch.tensor([target_token], device=device)
                loss += loss_fn(last_logits, target)

                next_tok = target.unsqueeze(0)
                context = torch.cat([context, next_tok], dim=1)

            loss = loss / len(tgt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}/{epochs} — loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "data/model.pt")
    print("Модель сохранена в data/model.pt")

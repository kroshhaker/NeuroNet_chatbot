import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import numpy as np
from model import TextRNN

class TextGenerator:
    def __init__(self, model_path="../data/model_rnn.pt", vocab_path="../data/vocab.pkl", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загрузка словаря
        with open(vocab_path, "rb") as f:
            data = pickle.load(f)

        stoi_raw = data['stoi']
        itos_raw = data['itos']

        max_index = max(itos_raw.values())
        self.itos = [None] * (max_index + 1)
        for c, i in itos_raw.items():
            self.itos[i] = c

        self.stoi = {}
        for c, i in stoi_raw.items():
            self.stoi[c] = i

        self.vocab_size = max_index + 1

        # Загрузка модели
        self.model = TextRNN(
            input_size=self.vocab_size,
            hidden_size=256,
            embedding_size=128,
            n_layers=2
        ).to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Модель успешно загружена")
        else:
            print("Файл модели не найден")
        self.model.eval()

    def encode(self, s):
        return [self.stoi.get(c, self.stoi["<UNK>"]) for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l if i < len(self.itos)])

    def generate(
            self,
            start_text,
            max_length=200,
            temperature=1.0,
            top_k=None,
            seed=None
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        tokens = self.encode(start_text)
        hidden = self.model.init_hidden(1)

        # Обработка начального контекста
        if len(tokens) > 1:
            for i in range(len(tokens) - 1):
                input_tensor = torch.tensor([[tokens[i]]], dtype=torch.long, device=self.device)
                _, hidden = self.model(input_tensor, hidden)

        input_tensor = torch.tensor([[tokens[-1]]], dtype=torch.long, device=self.device)
        generated = tokens.copy()

        for _ in range(max_length):
            with torch.no_grad():
                output, hidden = self.model(input_tensor, hidden)

            logits = output.squeeze() / temperature

            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                values, indices = torch.topk(logits, top_k)
                logits[logits < values[-1]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

        return self.decode(generated[len(tokens):])

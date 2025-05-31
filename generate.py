import torch
import pickle
import os

from model import TransformerModel


class TextGenerator:
    def __init__(self, model_path="data/model.pt", vocab_path="data/vocab.pkl", block_size=256, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.block_size = block_size

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

        assert all(self.itos[self.stoi[c]] == c for c in self.stoi)

        self.vocab_size = len(self.itos)

        self.load_model(model_path)

    def encode(self, s):
        return [self.stoi.get(c, self.stoi["<UNK>"]) for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l if i < len(self.itos)])

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели '{model_path}' не найден. Сначала обучите модель.")
        self.model = TransformerModel(self.vocab_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def generate(
            self,
            start_text,
            max_new_tokens=200,
            temperature=1.0,
            top_k=None,
            seed=None
    ):
        """
        Генерация текста

        Параметры:
        start_text: начальный текст для генерации
        max_new_tokens: максимальное количество новых токенов
        temperature: контролирует случайность (меньше = предсказуемее)
        top_k: ограничивает выбор топ-k наиболее вероятных токенов
        seed: фиксирует случайность для воспроизводимости
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        start_tokens = self.encode(start_text)
        context = start_tokens[-self.block_size:]
        generated = []

        for _ in range(max_new_tokens):
            x = torch.tensor([context], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.model(x)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                values, indices = torch.topk(logits, top_k)
                logits[logits < values[:, -1]] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            context.append(next_token)
            if len(context) > self.block_size:
                context = context[1:]

        generated_text = self.decode(generated)
        return generated_text
# Основные спецтокены
special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<SEP>', '<UNK>']

# Русский алфавит
ru = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" + "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"

# Английский алфавит
en = "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Цифры
digits = "0123456789"

# Общая пунктуация
punct = " .,!?;:-_()[]{}\"'`~@#$%^&*+=/\\|<>…«»—–·©®™"

# Управляющие
controls = "\n\t\r"

# Латинские буквы с диакритикой (Latin-1 Supplement)
latin1 = "ÀÁÂÃÄÅàáâãäåÈÉÊËèéêëÌÍÎÏìíîïÒÓÔÕÖØòóôõöøÙÚÛÜùúûüÝýÿÑñÇç"

# Греческий алфавит
greek = "αβγδεζηθικλμνξοπρστυφχψω" + "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"

# Дополнительные кириллические (Украинский, Белорусский, Сербский)
cyr_ext = "ґҐєЄіІїЇѐЁ"

# arabic = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
# hebrew = "אבגדהוזחטיכלמנסעפצקרשת"

# Объединяем всё
tokens = special_tokens + list(controls + punct + digits + en + latin1 + ru + greek + cyr_ext)

# Генерируем словари
stoi = {ch: i for i, ch in enumerate(tokens)}
itos = {i: ch for ch, i in enumerate(tokens)}

vocab = {"stoi": stoi, "itos": itos}

import pickle
with open("../data/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print(f"Словарь расширен до {len(tokens)} токенов и сохранён в data/vocab.pkl")
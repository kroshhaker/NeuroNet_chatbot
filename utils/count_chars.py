def start_counting():
    with open("../data/data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    num_chars = len(text)
    print(f"Количество символов в файле data.txt: {num_chars}")

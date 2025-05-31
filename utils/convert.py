import pickle
import json
import os

json_file_path = '../data/vocab.json'
pkl_file_path = '../data/vocab.pkl'

def toJson():
    with open(pkl_file_path, "rb") as f:
        vocab = pickle.load(f)
    with open(json_file_path, "w", encoding='utf-8') as f_json:
        json.dump(vocab, f_json, ensure_ascii=False, indent=4)

def toPkl():
    with open(json_file_path, 'r', encoding='utf-8') as f_json:
        data = json.load(f_json)
    with open(pkl_file_path, 'wb') as f_pkl:
        pickle.dump(data, f_pkl)

print("0 - exit")
print("1 - to Json")
print("2 - to Pkl")

mode = "0"
mode = input("Введите режим > ")
if mode == "0":
    pass
elif mode == "1":
    toJson()
elif mode == "2":
    toPkl()
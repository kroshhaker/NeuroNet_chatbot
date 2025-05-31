import pickle

with open("../data/vocab.pkl", "rb") as f:
    data = pickle.load(f)
print(type(data))
print(data)

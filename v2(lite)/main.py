from window import TextGeneratorWindow
from train import train_model

print("ChatBot NeuroNet Lite")
print("")
print("0 - exit")
print("1 - train")
print("2 - train_qa")
print("3 - run")
print("4 - utils")

while True:
    mode = 0
    mode = input("Select mode > ")
    if mode == "0":
        print("Goodbye")
        break
    elif mode == "1":
        train_model()
    #elif mode == "2":
    #    train_qa_model()
    elif mode == "3":
         app = TextGeneratorWindow()
         app.run()
    elif mode == "4":
        print("Soon")

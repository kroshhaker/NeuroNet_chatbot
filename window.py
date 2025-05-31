import tkinter as tk
from tkinter import scrolledtext
from generate import TextGenerator

class TextGeneratorWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Текстовая нейросеть")
        self.root.geometry("600x500")
        self.root.configure(bg="#f5f5f5")
        self.gen = TextGenerator()

        tk.Label(self.root, text="Введите текст:", bg="#f5f5f5", font=("Arial", 12)).pack(pady=(10, 0))

        self.input_field = tk.Entry(self.root, font=("Arial", 12), width=50)
        self.input_field.pack(pady=5)

        control_frame = tk.Frame(self.root, bg="#f5f5f5")
        control_frame.pack(pady=5, fill=tk.X, padx=60)

        tk.Label(control_frame, text="Токены:", bg="#f5f5f5", font=("Arial", 12)).pack(side=tk.LEFT, padx=(0, 10))
        self.tokens = tk.Entry(control_frame, font=("Arial", 12), width=10)
        self.tokens.pack(side=tk.LEFT)

        self.generate_btn = tk.Button(control_frame, text="Отправить", font=("Arial", 12), command=self.on_generate)
        self.generate_btn.pack(side=tk.RIGHT, padx=(0, 20))

        tk.Label(self.root, text="Полученный текст:", bg="#f5f5f5", font=("Arial", 12)).pack(pady=(20, 0))

        self.output_field = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=70, height=15,
                                                      font=("Consolas", 11), bg="#ffffff")
        self.output_field.pack(pady=5)
        self.output_field.config(state='disabled')

    def on_generate(self):
        start_text: str = self.input_field.get()

        result = self.generate_text(start_text)
        self.output_field.config(state='normal')
        self.output_field.delete("1.0", tk.END)
        self.output_field.insert(tk.END, result)
        self.output_field.config(state='disabled')

    def generate_text(self, start_text):
        output_text = self.gen.generate(start_text, max_new_tokens=int(self.tokens.get()))
        return output_text

    def run(self):
        self.root.mainloop()

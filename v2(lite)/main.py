from window import TextGeneratorWindow
from train import train_model
from train_qa import train_model as train_qa_model

from colorama import init, Fore, Style
init(autoreset=True)

# ASCII-арт заголовок
ascii_art = f"""{Fore.CYAN}
   ██████╗██╗  ██╗ █████╗ ████████╗   ███╗   ██╗███╗   ██╗
  ██╔════╝██║  ██║██╔══██╗╚══██╔══╝   ████╗  ██║████╗  ██║
  ██║     ███████║███████║   ██║      ██╔██╗ ██║██╔██╗ ██║
  ██║     ██╔══██║██╔══██║   ██║      ██║╚██╗██║██║╚██╗██║
  ╚██████╗██║  ██║██║  ██║   ██║      ██║ ╚████║██║ ╚████║
   ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝      ╚═╝  ╚═══╝╚═╝  ╚═══╝  
             {Style.BRIGHT + Fore.MAGENTA}Lite версия чат-бота NeuroNet
"""

print(ascii_art)
print(Style.BRIGHT + Fore.GREEN + "Выберите режим работы:")
print(Fore.BLUE + "0 - Выход")
print("1 - Обучить основную модель")
print("2 - Обучить модель Вопрос-Ответ")
print("3 - Запустить генератор текста")
print("4 - Утилиты (в разработке)\n")

while True:
    mode = input(Fore.CYAN + "Введите номер режима > " + Style.RESET_ALL)
    if mode == "0":
        print(Fore.RED + "До свидания!")
        break
    elif mode == "1":
        print(Fore.YELLOW + "Обучение основной модели...")
        train_model()
    elif mode == "2":
        print(Fore.YELLOW + "Обучение модели вопрос-ответ...")
        train_qa_model()
    elif mode == "3":
        print(Fore.YELLOW + "Запуск генератора текста...")
        app = TextGeneratorWindow()
        app.run()
    elif mode == "4":
        print(Fore.YELLOW + "Раздел 'Утилиты' скоро будет доступен.")
    else:
        print(Fore.RED + "Неверный ввод. Пожалуйста, выберите от 0 до 4.")

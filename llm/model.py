#!/usr/bin/env python
from gpt4all import GPT4All


def main():
    model = GPT4All("gpt4all-falcon-newbpe-q4_0.gguf")

    print("🤖 Локальный чат с LLM (выйти — напиши 'exit')")
    with model.chat_session() as session:
        while True:
            user_input = input("👤 Ты: ")
            if user_input.strip().lower() in ["exit", "quit", "выход"]:
                print("🛑 Завершение работы.")
                break

            response = model.generate(
                user_input,
                max_tokens=200,   # длина ответа
                temp=0.7,         # "креативность"
            )
            print("🤖 Модель:", response)


if __name__ == "__main__":
    main()

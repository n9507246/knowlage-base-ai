import os
import openai
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

load_dotenv()

class YandexGPT:
    """
    Класс-адаптер для взаимодействия с YandexGPT.
    Преобразует высокоуровневые объекты сообщений LangChain в формат, 
    совместимый с OpenAI API, который используется в Yandex Cloud.
    """

    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.getenv("YANDEX_CLOUD_API_KEY"),
            base_url="https://llm.api.cloud.yandex.net/v1",
            default_headers={
                "x-folder-id": os.getenv("YANDEX_CLOUD_FOLDER")
            },
        )

        self.model_path = (
            f"gpt://{os.getenv('YANDEX_CLOUD_FOLDER')}/yandexgpt/latest"
        )

    def _convert_messages(self, messages) -> list[dict]:
        """
        Конвертирует любые форматы сообщений в формат словарей для YandexGPT.
        Поддерживает: списки, кортежи, отдельные сообщения, строки, словари, BaseMessage
        """
        formatted_messages = []
        
        # Если пришел не список, а кортеж или одиночное сообщение
        if not isinstance(messages, (list, tuple)):
            messages = [messages]
        
        # Если пришел кортеж, конвертируем в список
        if isinstance(messages, tuple):
            messages = list(messages)
        
        for msg in messages:
            # Если сообщение уже является словарем
            if isinstance(msg, dict):
                # Проверяем, что словарь имеет нужные ключи
                if 'role' in msg and 'content' in msg:
                    formatted_messages.append(msg)
                else:
                    # Пытаемся извлечь content из произвольного словаря
                    content = msg.get('content') or msg.get('text') or msg.get('message') or str(msg)
                    formatted_messages.append({
                        "role": msg.get('role', 'user'),
                        "content": content
                    })
                continue
            
            # Если сообщение - строка
            if isinstance(msg, str):
                formatted_messages.append({
                    "role": "user",
                    "content": msg
                })
                continue
            
            # Если сообщение - BaseMessage (HumanMessage, AIMessage, SystemMessage)
            if isinstance(msg, BaseMessage):
                # Определяем роль
                if isinstance(msg, SystemMessage):
                    role = "system"
                elif isinstance(msg, AIMessage):
                    role = "assistant"
                else:  # HumanMessage и все остальные
                    role = "user"
                    
                formatted_messages.append({
                    "role": role,
                    "content": msg.content
                })
                continue
            
            # Если ничего не подошло, конвертируем в строку
            formatted_messages.append({
                "role": "user",
                "content": str(msg)
            })
            
        return formatted_messages

    def ask(
        self,
        messages,  # убираем аннотацию типа для гибкости
        max_tokens: int = 2000,
        temp: float = 0.3
    ) -> str:
        """
        Основной метод для генерации ответа.
        Принимает сообщения в любом формате и отправляет запрос в облако.
        """

        # 1. Приводим сообщения к единому формату словарей
        api_messages = self._convert_messages(messages)
        
        # 2. Выполняем HTTP-запрос
        try:
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=temp,
            )
            return response 
        except Exception as e:
            print(f"❌ Ошибка YandexGPT API: {e}")
            raise
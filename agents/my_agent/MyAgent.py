from rich.console import Console
from rich.panel import Panel


class MyAgent:
    
    def __init__(self, knowledge_base_path: str = "./knowledge_base"):
        self.knowledge_base_path = knowledge_base_path
        self.console = Console()


    def ask(self, query, clarify=True):
        """Метод для обработки запросов пользователя"""
        return "🤖 Ответ от LLM: Этот функционал еще не реализован"

from rich.console import Console
from rich.panel import Panel


class MyAgent:
    
    def __init__(self, knowledge_base_path: str = "./knowledge_base"):
        self.knowledge_base_path = knowledge_base_path
        self.console = Console()

    def out_message(self):
        """Метод для вывода сообщения о завершении работы"""
        self.console.print("[red]Завершение работы ассистента[/red]")

    def ask(self, query, clarify=True):
        """Метод для обработки запросов пользователя"""
        self.console.print(f"📝 Вопрос: {query}")
        self.console.print(f"🔍 Уточнение: {clarify}")
        self.console.print("🤖 Ответ от LLM: Этот функционал еще не реализован")

    def welcome_message(self):
        """Метод для приветственного сообщения"""
        self.console.print(Panel.fit(
            "🛠️ [bold cyan]Ассистент системного администратора[/bold cyan]\n"
            "Задавайте вопросы о настройке, устранении неисправностей и автоматизации.",
            style="blue"
        ))
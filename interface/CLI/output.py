from rich.console import Console as RichConsole
from rich.panel import Panel

class Output:
    
    outputCli = RichConsole()

    @staticmethod
    def welcome_message():
        Output.outputCli.print(Panel.fit(
            "🛠️ [bold cyan]Ассистент системного администратора[/bold cyan]\n"
            "Задавайте вопросы о настройке, устранении неисправностей и автоматизации.",
            style="blue"
        ))
    
    @staticmethod
    def print_answer( answer, clarify=True):

        Output.outputCli.print(f"AI: {answer}")

    @staticmethod
    def out_message():
        Output.outputCli.print("\n[yellow]Завершение работы...[/yellow]")

    @staticmethod
    def err_message(e: Exception):
        Output.outputCli.print(f"[red]Ошибка: {e}[/red]")

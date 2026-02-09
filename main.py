from agents.my_agent.MyAgent import MyAgent

# Пример использования
def main():
    agent = MyAgent(knowledge_base_path="./knowledge_base")
    
    agent.welcome_message()
    
    while True:
        try:
            query = input("\n🔍 Ваш вопрос (или 'quit' для выхода): ").strip()
            
            if query.lower() in ['quit', 'exit', 'выход']:
                agent.out_message()
                break
            
            if query:
                agent.ask(query, clarify=True)
                
        except KeyboardInterrupt:
            agent.console.print("\n[yellow]Завершение работы...[/yellow]")
            agent.out_message()
            break
        except Exception as e:
            agent.console.print(f"[red]Ошибка: {e}[/red]")

if __name__ == "__main__":
    main()
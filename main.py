from agents.my_agent.MyAgent import MyAgent
from output.Console import Console

# Пример использования
def main():
    agent = MyAgent(knowledge_base_path="./knowledge_base")
    
    Console.welcome_message()    
    
    while True:
        try:
            query = input("\n🔍 Ваш вопрос (или 'quit' для выхода): ").strip()
            
            if query.lower() in ['quit', 'exit', 'выход']:
                Console.out_message()
                break
            
            if query:

                result = agent.run(query, clarify=True)

                Console.print_answer(
                    result.get("output")
                )
                
        except KeyboardInterrupt:
            Console.out_message()
            break
        except Exception as e:
            Console.err_message(e)

if __name__ == "__main__":
    main()
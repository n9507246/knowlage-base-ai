from .output import Output

class CLI:

    @staticmethod
    def run(agent):
        """Запуск CLI интерфейса"""
        Output.welcome_message()
        
        while True:
            try:
                query = input("\n🔍 Ваш вопрос (или 'quit' для выхода): ").strip()
                
                if query.lower() in ['quit', 'exit', 'выход']:
                    Output.out_message()
                    break
                
                if query:
                    result = agent.run(query)
                    
                    if isinstance(result, dict):
                        answer = result.get("output") or result.get("answer") or result.get("response") or str(result)
                    else:
                        answer = str(result)
                    
                    Output.print_answer(answer)
                    
            except KeyboardInterrupt:
                Output.out_message()
                break
            except Exception as e:
                Output.err_message(f"Ошибка: {str(e)}")
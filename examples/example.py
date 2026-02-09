import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import print as rprint

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ваш YandexGPT адаптер
from your_module import YandexGPT

class SysAdminAssistant:
    def __init__(self, knowledge_base_path: str = "./knowledge_base"):
        self.console = Console()
        self.knowledge_base_path = knowledge_base_path
        self.log_file = "assistant_log.txt"
        
        # Инициализация LLM
        self.llm = YandexGPT()
        
        # Память для диалога (для уточняющих вопросов)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Инициализация или загрузка векторной БД
        self.vectorstore = self._initialize_vectorstore()
        self.retriever = self._create_retriever()
        self.qa_chain = self._create_qa_chain()
        
        # Статистика
        self.stats = {
            "queries": 0,
            "found_answers": 0,
            "not_found": 0,
            "clarifications": 0
        }
    
    def _initialize_vectorstore(self):
        """Инициализация ChromaDB с индексацией документов"""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            # Эта модель поддерживает русский язык
        )
        
        persist_directory = "./chroma_db"
        
        # Проверяем, существует ли уже векторная БД
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            self.console.print("[green]Загружаю существующую базу знаний...[/green]")
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
        
        # Индексация новых документов
        self.console.print("[yellow]Индексирую базу знаний...[/yellow]")
        
        # Загрузка всех .md файлов
        loader = DirectoryLoader(
            self.knowledge_base_path,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        
        documents = loader.load()
        
        if not documents:
            self.console.print("[red]В базе знаний нет документов![/red]")
            # Создаем пустую БД
            return Chroma.from_documents(
                documents=[],  # пустой список
                embedding=embeddings,
                persist_directory=persist_directory
            )
        
        # Разбиение на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Добавляем метаданные
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get('source', 'unknown')
            # Извлекаем категорию из пути
            relative_path = os.path.relpath(source, self.knowledge_base_path)
            category = os.path.dirname(relative_path)
            
            chunk.metadata.update({
                "chunk_index": i,
                "category": category,
                "indexed_date": datetime.now().isoformat(),
                "file_name": os.path.basename(source)
            })
        
        self.console.print(f"[green]Проиндексировано {len(chunks)} фрагментов из {len(documents)} документов[/green]")
        
        # Создание векторной БД
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        return vectorstore
    
    def _create_retriever(self):
        """Создание ретривера с возможностью переконфигурации"""
        return self.vectorstore.as_retriever(
            search_kwargs={
                "k": 4,
                # Можно добавить фильтр: "filter": {"category": "linux"}
            }
        )
    
    def _create_qa_chain(self):
        """Создание цепочки вопрос-ответ с кастомным промптом"""
        
        prompt_template = """Ты — ассистент системного администратора. Твоя задача — помогать с техническими вопросами на основе базы знаний.

Контекст из базы знаний:
{context}

История диалога:
{chat_history}

Вопрос: {question}

Инструкции:
1. Отвечай ТОЛЬКО на основе предоставленного контекста
2. Если информации в контексте недостаточно для ответа, скажи: "В базе знаний нет информации по этому вопросу."
3. Если нужно уточнение, задай clarifying вопрос
4. Форматируй ответ с использованием markdown для читаемости
5. Если есть пошаговая инструкция, представь её в виде нумерованного списка
6. В конце укажи источники (если они есть)

Ответ:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Создаем цепочку, совместимую с вашим YandexGPT
        from langchain.chains import LLMChain
        from langchain.callbacks.base import BaseCallbackHandler
        
        class CustomRetrievalQA:
            def __init__(self, llm, retriever, prompt, memory):
                self.llm = llm
                self.retriever = retriever
                self.prompt = prompt
                self.memory = memory
            
            def invoke(self, query_dict):
                # Получаем релевантные документы
                docs = self.retriever.get_relevant_documents(query_dict["query"])
                
                # Формируем контекст
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Получаем историю диалога
                chat_history = self.memory.load_memory_variables({})["chat_history"]
                history_text = "\n".join([msg.content for msg in chat_history])
                
                # Формируем промпт
                full_prompt = self.prompt.format(
                    context=context,
                    chat_history=history_text,
                    question=query_dict["query"]
                )
                
                # Отправляем в LLM
                response = self.llm.ask(
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=2000,
                    temp=0.3
                )
                
                # Сохраняем в память
                self.memory.save_context(
                    {"input": query_dict["query"]},
                    {"output": response}
                )
                
                return {
                    "result": response,
                    "source_documents": docs,
                    "context": context
                }
        
        return CustomRetrievalQA(
            llm=self.llm,
            retriever=self.retriever,
            prompt=prompt,
            memory=self.memory
        )
    
    def _log_interaction(self, query: str, response: str, sources: List, found: bool):
        """Логирование взаимодействия"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            log_entry = f"""
            --- {datetime.now().isoformat()} ---
            Вопрос: {query}
            
            Ответ: {response}
            
            Найдено источников: {len(sources)}
            Ответ найден: {found}
            
            Источники:
            {chr(10).join([f'- {doc.metadata.get("source", "N/A")}' for doc in sources])}
            {'='*50}
            """
            f.write(log_entry)
    
    def _display_sources(self, sources: List):
        """Красивое отображение источников"""
        if not sources:
            return
        
        table = Table(title="Источники информации")
        table.add_column("Файл", style="cyan")
        table.add_column("Категория", style="green")
        table.add_column("Релевантность", style="yellow")
        
        for doc in sources:
            source = doc.metadata.get("file_name", "unknown")
            category = doc.metadata.get("category", "N/A")
            # Можно добавить оценку релевантности, если ChromaDB её возвращает
            relevance = "высокая"  # Заглушка
            
            table.add_row(source, category, relevance)
        
        self.console.print(table)
    
    def ask(self, query: str, clarify: bool = True) -> Optional[str]:
        """Основной метод для вопросов"""
        self.stats["queries"] += 1
        
        # Логируем вопрос
        self.console.print(Panel(f"❓ [bold]Вопрос:[/bold] {query}", style="blue"))
        
        # Получаем ответ
        result = self.qa_chain.invoke({"query": query})
        
        # Проверяем, есть ли полезная информация
        sources = result.get("source_documents", [])
        answer = result.get("result", "")
        
        # Анализ ответа
        not_found_phrases = [
            "нет информации",
            "не смог найти",
            "не содержится в базе",
            "не указано в базе"
        ]
        
        found = all(phrase not in answer.lower() for phrase in not_found_phrases)
        
        if found:
            self.stats["found_answers"] += 1
            self.console.print(Panel(
                Markdown(answer),
                title="🤖 Ответ ассистента",
                style="green"
            ))
        else:
            self.stats["not_found"] += 1
            self.console.print(Panel(
                "[yellow]В базе знаний нет информации по этому вопросу.[/yellow]\n"
                "Хотите:\n"
                "1. Переформулировать запрос\n"
                "2. Уточнить вопрос\n"
                "3. Пропустить",
                title="⚠️ Информация не найдена",
                style="yellow"
            ))
        
        # Показываем источники
        if sources:
            self._display_sources(sources)
        
        # Логируем
        self._log_interaction(query, answer, sources, found)
        
        # Предлагаем уточнить или переформулировать
        if not found and clarify:
            return self._clarify_loop(query)
        
        return answer if found else None
    
    def _clarify_loop(self, original_query: str) -> Optional[str]:
        """Цикл уточнения вопроса"""
        while True:
            self.console.print("\n[cyan]Что вы хотите сделать?[/cyan]")
            self.console.print("1. Переформулировать запрос")
            self.console.print("2. Задать уточняющий вопрос")
            self.console.print("3. Вернуться к поиску")
            self.console.print("4. Выйти")
            
            choice = input("\nВаш выбор (1-4): ").strip()
            
            if choice == "1":
                new_query = input("Введите новый запрос: ").strip()
                if new_query:
                    return self.ask(new_query, clarify=True)
            
            elif choice == "2":
                # Ассистент пытается уточнить
                clarification_prompt = f"""
                Пользователь задал вопрос: "{original_query}"
                В базе знаний не найдено точной информации.
                Сформулируй уточняющий вопрос, который поможет найти нужную информацию.
                Будь конкретен и предложи варианты уточнения.
                """
                
                clarification = self.llm.ask(
                    messages=[{"role": "user", "content": clarification_prompt}],
                    max_tokens=300
                )
                
                self.console.print(Panel(
                    clarification,
                    title="🤔 Уточняющий вопрос",
                    style="cyan"
                ))
                
                user_response = input("Ваш ответ на уточнение: ").strip()
                if user_response:
                    combined_query = f"{original_query} {user_response}"
                    return self.ask(combined_query, clarify=True)
            
            elif choice == "3":
                return None
            
            elif choice == "4":
                self.show_stats()
                exit(0)
    
    def add_document(self, file_path: str):
        """Добавление нового документа в базу знаний"""
        if not file_path.endswith('.md'):
            self.console.print("[red]Поддерживаются только .md файлы[/red]")
            return
        
        # Копируем файл в базу знаний
        import shutil
        target_path = os.path.join(self.knowledge_base_path, os.path.basename(file_path))
        shutil.copy2(file_path, target_path)
        
        # Переиндексируем
        self.console.print("[yellow]Переиндексирую базу знаний...[/yellow]")
        # Здесь нужно реализовать переиндексацию
        # Можно пересоздать векторную БД или добавить только новый документ
    
    def show_stats(self):
        """Показать статистику работы"""
        table = Table(title="Статистика работы ассистента")
        table.add_column("Метрика", style="cyan")
        table.add_column("Значение", style="green")
        
        table.add_row("Всего запросов", str(self.stats["queries"]))
        table.add_row("Найдено ответов", str(self.stats["found_answers"]))
        table.add_row("Не найдено", str(self.stats["not_found"]))
        table.add_row("Уточнений", str(self.stats["clarifications"]))
        
        if self.stats["queries"] > 0:
            success_rate = (self.stats["found_answers"] / self.stats["queries"]) * 100
            table.add_row("Успешность", f"{success_rate:.1f}%")
        
        self.console.print(table)

# Пример использования
def main():
    assistant = SysAdminAssistant(knowledge_base_path="./knowledge_base")
    
    # Интерактивный режим
    assistant.console.print(Panel.fit(
        "🛠️ [bold cyan]Ассистент системного администратора[/bold cyan]\n"
        "Задавайте вопросы о настройке, устранении неисправностей и автоматизации.",
        style="blue"
    ))
    
    while True:
        try:
            query = input("\n🔍 Ваш вопрос (или 'quit' для выхода): ").strip()
            
            if query.lower() in ['quit', 'exit', 'выход']:
                assistant.show_stats()
                break
            
            if query:
                assistant.ask(query, clarify=True)
                
        except KeyboardInterrupt:
            assistant.console.print("\n[yellow]Завершение работы...[/yellow]")
            assistant.show_stats()
            break
        except Exception as e:
            assistant.console.print(f"[red]Ошибка: {e}[/red]")

if __name__ == "__main__":
    main()
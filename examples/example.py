import os
from typing import TypedDict, Annotated, Sequence, Optional
from datetime import datetime
from enum import Enum
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import print as rprint

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Ваш YandexGPT адаптер
from your_module import YandexGPT

# ==================== ОПРЕДЕЛЕНИЕ СОСТОЯНИЯ ====================

class AgentState(TypedDict):
    """Состояние агента в графе"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    retrieved_docs: list
    needs_clarification: bool
    clarification_question: str
    search_performed: bool
    current_topic: str
    response: Optional[str]
    sources: list

# ==================== КЛАСС ДЛЯ РАБОТЫ С БАЗОЙ ЗНАНИЙ ====================

class KnowledgeBaseManager:
    def __init__(self, knowledge_base_path: str = "./knowledge_base"):
        self.knowledge_base_path = knowledge_base_path
        
        # Модель для эмбеддингов (поддерживает русский)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        self.persist_directory = "./chroma_db"
        self.vectorstore = self._init_vectorstore()
    
    def _init_vectorstore(self):
        """Инициализация векторного хранилища"""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        
        # Индексация документов
        console = Console()
        console.print("[yellow]Индексирую базу знаний...[/yellow]")
        
        loader = DirectoryLoader(
            self.knowledge_base_path,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        
        documents = loader.load()
        
        if not documents:
            console.print("[red]База знаний пуста![/red]")
            return Chroma.from_documents(
                documents=[],
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        
        # Умное чанкирование с сохранением структуры
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Обогащаем метаданные
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get('source', 'unknown')
            relative_path = os.path.relpath(source, self.knowledge_base_path)
            category = os.path.dirname(relative_path)
            
            # Извлекаем заголовок из содержимого
            content = chunk.page_content
            first_line = content.split('\n')[0] if '\n' in content else content[:100]
            
            chunk.metadata.update({
                "chunk_id": f"{os.path.basename(source)}_chunk_{i}",
                "category": category,
                "indexed_date": datetime.now().isoformat(),
                "title": first_line[:200],
                "source_file": os.path.basename(source)
            })
        
        console.print(f"[green]Создано {len(chunks)} фрагментов[/green]")
        
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def search(self, query: str, k: int = 4, filters: Optional[dict] = None) -> tuple[list, list]:
        """Поиск в базе знаний"""
        if filters:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={
                    "k": k,
                    "filter": filters
                }
            )
        else:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        docs = retriever.get_relevant_documents(query)
        
        # Извлекаем содержимое и метаданные
        contents = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        
        return contents, metadatas
    
    def add_document(self, file_path: str):
        """Добавление документа (упрощенное)"""
        # Здесь можно реализовать инкрементальное добавление
        # Для простоты переиндексируем всё
        self._init_vectorstore()

# ==================== УЗЛЫ ГРАФА ====================

class SysAdminAgent:
    def __init__(self):
        self.console = Console()
        self.kb_manager = KnowledgeBaseManager()
        self.llm = YandexGPT()
        
        # Промпты для разных задач
        self.system_prompt = """Ты — ассистент системного администратора. 
        Твоя задача — помогать с техническими вопросами на основе базы знаний.
        
        Правила:
        1. Отвечай ТОЛЬКО на основе предоставленного контекста из базы знаний
        2. Если информации недостаточно, попроси уточнить или скажи что не знаешь
        3. Форматируй ответ в markdown
        4. Будь краток, но информативен
        5. Предлагай конкретные команды и решения"""
        
        self.clarification_prompt = """Ты должен задать уточняющий вопрос пользователю.
        Вопрос должен быть конкретным и помогающим сузить поиск.
        
        Контекст запроса: {query}
        
        Сформулируй один уточняющий вопрос."""
        
        self.router_prompt = """Определи тип запроса пользователя.
        
        Типы запросов:
        1. ТЕХНИЧЕСКИЙ_ВОПРОС - вопрос о настройке, устранении неисправностей и т.д.
        2. УТОЧНЕНИЕ - ответ на уточняющий вопрос ассистента
        3. ПЕРЕФОРМУЛИРОВКА - пользователь переформулирует предыдущий запрос
        4. НОВАЯ_ТЕМА - совершенно новый вопрос
        
        Запрос: {query}
        
        Верни ТОЛЬКО тип запроса из списка выше."""
    
    # ==================== ОСНОВНЫЕ УЗЛЫ ====================
    
    def route_query(self, state: AgentState) -> str:
        """Определяет тип запроса и маршрутизирует"""
        query = state["query"]
        
        response = self.llm.ask(
            messages=[
                {"role": "system", "content": self.router_prompt.format(query=query)},
                {"role": "user", "content": query}
            ],
            max_tokens=50,
            temp=0.1
        )
        
        # Чистим ответ
        response = response.strip().upper()
        
        if "ТЕХНИЧЕСКИЙ_ВОПРОС" in response or "НОВАЯ_ТЕМА" in response:
            return "search_knowledge_base"
        elif "УТОЧНЕНИЕ" in response:
            return "handle_clarification"
        elif "ПЕРЕФОРМУЛИРОВКА" in response:
            return "reformulate_and_search"
        else:
            return "search_knowledge_base"
    
    def search_knowledge_base(self, state: AgentState) -> dict:
        """Поиск в базе знаний"""
        query = state["query"]
        
        # Пытаемся определить категорию для фильтрации
        category_filters = self._infer_category(query)
        
        # Поиск
        contents, metadatas = self.kb_manager.search(
            query=query,
            k=4,
            filters=category_filters
        )
        
        # Формируем контекст
        context = self._format_context(contents, metadatas)
        
        # Проверяем, достаточно ли информации
        needs_clarification = len(contents) == 0 or self._is_vague_context(contents)
        
        if needs_clarification:
            clarification = self._generate_clarification(query, contents)
            return {
                "retrieved_docs": contents,
                "needs_clarification": True,
                "clarification_question": clarification,
                "search_performed": True
            }
        else:
            return {
                "retrieved_docs": contents,
                "needs_clarification": False,
                "search_performed": True
            }
    
    def generate_response(self, state: AgentState) -> dict:
        """Генерация ответа на основе найденной информации"""
        query = state["query"]
        docs = state.get("retrieved_docs", [])
        
        if not docs or len(docs) == 0:
            response = "Извините, в базе знаний нет информации по этому вопросу."
            
            # Предлагаем варианты действий
            response += "\n\n**Что можно сделать:**"
            response += "\n1. Переформулировать запрос"
            response += "\n2. Добавить информацию в базу знаний"
            response += "\n3. Задать более конкретный вопрос"
            
            return {
                "response": response,
                "sources": []
            }
        
        # Формируем промпт с контекстом
        context = "\n\n".join([doc for doc in docs[:3]])  # Берем топ-3
        
        prompt = f"""{self.system_prompt}

Контекст из базы знаний:
{context}

Вопрос пользователя: {query}

Сформулируй ответ на основе контекста выше. Если в контексте нет точной информации, скажи об этом.
В конце укажи, на каких источниках основан ответ."""

        response = self.llm.ask(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temp=0.3
        )
        
        # Извлекаем источники из метаданных (в реальности нужно передавать их из состояния)
        sources = [f"Документ {i+1}" for i in range(min(3, len(docs)))]
        
        return {
            "response": response,
            "sources": sources
        }
    
    def ask_clarification(self, state: AgentState) -> dict:
        """Задаем уточняющий вопрос"""
        clarification = state.get("clarification_question", 
                                "Не могли бы вы уточнить ваш запрос?")
        
        return {
            "response": f"**Уточняющий вопрос:** {clarification}\n\nПожалуйста, уточните ваш запрос.",
            "needs_clarification": True
        }
    
    def handle_clarification_response(self, state: AgentState) -> dict:
        """Обработка ответа на уточняющий вопрос"""
        # Комбинируем оригинальный запрос с уточнением
        messages = state["messages"]
        
        # Находим последний вопрос ассистента и ответ пользователя
        last_assistant_msg = None
        last_user_msg = None
        
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and last_assistant_msg is None:
                last_assistant_msg = msg.content
            elif isinstance(msg, HumanMessage) and last_user_msg is None:
                last_user_msg = msg.content
            
            if last_assistant_msg and last_user_msg:
                break
        
        # Формируем улучшенный запрос
        if last_assistant_msg and last_user_msg:
            improved_query = f"{state['query']} {last_user_msg}"
            return {"query": improved_query, "needs_clarification": False}
        
        return {"needs_clarification": False}
    
    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================
    
    def _infer_category(self, query: str) -> Optional[dict]:
        """Определяет категорию запроса для фильтрации"""
        query_lower = query.lower()
        
        category_map = {
            "linux": ["linux", "ubuntu", "debian", "centos", "bash", "ssh"],
            "windows": ["windows", "powershell", "active directory", "ad"],
            "network": ["сеть", "network", "ip", "dns", "vpn", "firewall"],
            "docker": ["docker", "контейнер", "container", "docker-compose"],
            "samba": ["samba", "общая папка", "расшаренная папка"]
        }
        
        for category, keywords in category_map.items():
            if any(keyword in query_lower for keyword in keywords):
                return {"category": category}
        
        return None
    
    def _format_context(self, contents: list, metadatas: list) -> str:
        """Форматирует контекст для промпта"""
        context_parts = []
        
        for i, (content, metadata) in enumerate(zip(contents, metadatas)):
            source = metadata.get('source_file', f'Источник {i+1}')
            title = metadata.get('title', '')
            
            context_parts.append(f"--- {source} ---")
            if title:
                context_parts.append(f"Заголовок: {title}")
            context_parts.append(content)
            context_parts.append("")  # Пустая строка
        
        return "\n".join(context_parts)
    
    def _is_vague_context(self, contents: list) -> bool:
        """Определяет, является ли контекст слишком общим"""
        if not contents:
            return True
        
        # Проверяем длину контекста
        total_length = sum(len(content) for content in contents)
        if total_length < 500:  # Слишком мало информации
            return True
        
        # Проверяем разнообразие источников
        return False
    
    def _generate_clarification(self, query: str, docs: list) -> str:
        """Генерирует уточняющий вопрос"""
        prompt = self.clarification_prompt.format(query=query)
        
        # Добавляем контекст найденных документов
        if docs:
            context_preview = "\n".join([doc[:200] + "..." for doc in docs[:2]])
            prompt += f"\n\nНайдена следующая информация:\n{context_preview}"
        
        response = self.llm.ask(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temp=0.3
        )
        
        return response.strip()

# ==================== СОЗДАНИЕ ГРАФА ====================

def create_agent_graph() -> StateGraph:
    """Создает граф агента"""
    
    agent = SysAdminAgent()
    
    # Создаем граф
    workflow = StateGraph(AgentState)
    
    # Определяем узлы
    workflow.add_node("route_query", agent.route_query)
    workflow.add_node("search_knowledge_base", agent.search_knowledge_base)
    workflow.add_node("generate_response", agent.generate_response)
    workflow.add_node("ask_clarification", agent.ask_clarification)
    workflow.add_node("handle_clarification_response", agent.handle_clarification_response)
    
    # Начальная точка
    workflow.set_entry_point("route_query")
    
    # Определяем переходы из route_query
    workflow.add_conditional_edges(
        "route_query",
        agent.route_query,  # Функция возвращает имя следующего узла
        {
            "search_knowledge_base": "search_knowledge_base",
            "handle_clarification": "handle_clarification_response",
            "reformulate_and_search": "search_knowledge_base"
        }
    )
    
    # Переходы из search_knowledge_base
    def decide_after_search(state: AgentState) -> str:
        if state.get("needs_clarification", False):
            return "ask_clarification"
        else:
            return "generate_response"
    
    workflow.add_conditional_edges(
        "search_knowledge_base",
        decide_after_search,
        {
            "ask_clarification": "ask_clarification",
            "generate_response": "generate_response"
        }
    )
    
    # Переходы из handle_clarification_response
    workflow.add_edge("handle_clarification_response", "search_knowledge_base")
    
    # Переходы из ask_clarification
    workflow.add_edge("ask_clarification", END)
    
    # Переходы из generate_response
    workflow.add_edge("generate_response", END)
    
    # Добавляем контрольные точки для сохранения состояния
    memory = MemorySaver()
    
    # Компилируем граф
    graph = workflow.compile(checkpointer=memory)
    
    return graph, agent

# ==================== ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ ====================

class SysAdminAssistantCLI:
    def __init__(self):
        self.console = Console()
        self.graph, self.agent = create_agent_graph()
        self.current_thread = {"configurable": {"thread_id": "user_thread"}}
        
        self._print_welcome()
    
    def _print_welcome(self):
        """Приветственное сообщение"""
        welcome = Panel.fit(
            "[bold cyan]🤖 Ассистент системного администратора (LangGraph)[/bold cyan]\n\n"
            "Особенности:\n"
            "• Интеллектуальный поиск в базе знаний\n"
            "• Уточняющие вопросы при необходимости\n"
            "• Поддержка диалога с контекстом\n"
            "• Автоматическая категоризация запросов\n\n"
            "Команды:\n"
            "• /stats - статистика\n"
            "• /add <файл> - добавить документ\n"
            "• /clear - очистить историю\n"
            "• /quit - выход",
            style="blue"
        )
        self.console.print(welcome)
    
    def process_query(self, query: str):
        """Обработка запроса пользователя"""
        # Инициализируем состояние
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "retrieved_docs": [],
            "needs_clarification": False,
            "clarification_question": "",
            "search_performed": False,
            "current_topic": "",
            "response": None,
            "sources": []
        }
        
        # Выполняем граф
        try:
            result = self.graph.invoke(
                initial_state,
                config=self.current_thread
            )
            
            # Отображаем результат
            self._display_result(result)
            
        except Exception as e:
            self.console.print(f"[red]Ошибка: {e}[/red]")
    
    def _display_result(self, result: dict):
        """Отображение результата работы агента"""
        response = result.get("response", "")
        
        if not response:
            return
        
        # Определяем стиль панели
        if "Извините" in response or "нет информации" in response:
            style = "yellow"
            title = "⚠️ Информация не найдена"
        elif "Уточняющий вопрос" in response:
            style = "cyan"
            title = "🤔 Требуется уточнение"
        else:
            style = "green"
            title = "✅ Ответ"
        
        # Отображаем ответ
        self.console.print(Panel(
            Markdown(response),
            title=title,
            style=style
        ))
        
        # Отображаем источники, если есть
        sources = result.get("sources", [])
        if sources and "Уточняющий вопрос" not in response:
            self._display_sources(sources)
        
        # Логируем
        self._log_interaction(result)
    
    def _display_sources(self, sources: list):
        """Отображает источники информации"""
        table = Table(title="📚 Использованные источники")
        table.add_column("№", style="cyan")
        table.add_column("Тип", style="green")
        table.add_column("Описание", style="white")
        
        for i, source in enumerate(sources, 1):
            table.add_row(str(i), "Документ", source)
        
        self.console.print(table)
    
    def _log_interaction(self, result: dict):
        """Логирование взаимодействия"""
        log_file = "assistant_log.jsonl"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": result.get("query", ""),
            "response_preview": result.get("response", "")[:500],
            "sources_count": len(result.get("sources", [])),
            "needed_clarification": result.get("needs_clarification", False)
        }
        
        import json
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def run_interactive(self):
        """Интерактивный режим работы"""
        while True:
            try:
                query = input("\n🔍 Ваш вопрос: ").strip()
                
                if not query:
                    continue
                
                # Проверяем команды
                if query.lower() in ['/quit', '/exit', '/выход']:
                    self.console.print("[yellow]Завершение работы...[/yellow]")
                    break
                elif query.lower() == '/stats':
                    self._show_stats()
                    continue
                elif query.lower() == '/clear':
                    self._clear_history()
                    continue
                elif query.startswith('/add '):
                    self._add_document(query[5:].strip())
                    continue
                
                # Обрабатываем обычный запрос
                self.process_query(query)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Завершение работы...[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Ошибка: {e}[/red]")
    
    def _show_stats(self):
        """Показать статистику (заглушка)"""
        self.console.print("[cyan]Статистика будет реализована в будущем[/cyan]")
    
    def _clear_history(self):
        """Очистить историю диалога"""
        # Создаем новый тред
        import uuid
        self.current_thread = {"configurable": {"thread_id": f"thread_{uuid.uuid4()}"}}
        self.console.print("[green]История диалога очищена[/green]")
    
    def _add_document(self, file_path: str):
        """Добавить документ в базу знаний"""
        try:
            self.agent.kb_manager.add_document(file_path)
            self.console.print(f"[green]Документ добавлен: {file_path}[/green]")
        except Exception as e:
            self.console.print(f"[red]Ошибка при добавлении документа: {e}[/red]")

# ==================== ЗАПУСК ====================

def main():
    assistant = SysAdminAssistantCLI()
    assistant.run_interactive()

if __name__ == "__main__":
    main()
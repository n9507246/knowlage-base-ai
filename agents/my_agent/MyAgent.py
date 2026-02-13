from langgraph.graph import StateGraph, START, END
from node.LLM.LLM import LLM

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    input: str   # Входящий запрос от пользователя
    output: str  # Ответ, сформированный моделью

class MyAgent:
    
    def __init__(self, knowledge_base_path: str = "./knowledge_base"):
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("LLM", LLM(model=None, system_prompt=None))

        graph_builder.add_edge(START, "LLM")
        graph_builder.add_edge("LLM", END)

        self.graph = graph_builder.compile()

    def run(self, query, clarify=True):
        return self.graph.invoke({
            "input": "", 
            "output": "",
        })

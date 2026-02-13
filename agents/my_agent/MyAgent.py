from langgraph.graph import StateGraph, START, END
from node.LLM.LLM import LLM
from langchain_core.messages import HumanMessage, message_to_dict
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from YandexGPT import YandexGPT


class AgentState(TypedDict):
    input: HumanMessage   # Входящий запрос от пользователя
    output: str  # Ответ, сформированный моделью

class MyAgent:
    
    def __init__(self, knowledge_base_path: str = "./knowledge_base"):
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("LLM", LLM(model=YandexGPT(), system_prompt="Ты полезный ассистент"))

        graph_builder.add_edge(START, "LLM")
        graph_builder.add_edge("LLM", END)

        self.graph = graph_builder.compile()

    def run(self, user_query: str, clarify=True):

        result = self.graph.invoke({
            "input": HumanMessage(content=user_query), 
            "output": "",
        }).get("output")

        return result.choices[0].message.content
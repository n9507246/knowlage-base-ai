import uuid
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class LLM:
    def __init__(self, model, system_prompt):
        self.model = model
        self.system_prompt = system_prompt

    def __call__(self, state: dict) -> dict:

        user_query = state.get("input")

        return {
            "output": self.model.ask( user_query ),
        }

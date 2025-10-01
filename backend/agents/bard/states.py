from typing import List, Dict, Literal, Annotated, TypedDict, Any
from pydantic import BaseModel, Field
import operator


# ======================= Validation =======================


class Router(BaseModel):
    next_node: Literal["storyteller", "chat"]


class Query(BaseModel):
    query: List[str]
    max_results: int = 3


# ======================= States ==========================


class StoryTellerState(TypedDict):
    task: str
    age: Any
    retrieved_content: List[Dict]
    output: str
    next_node: str
    node_name: str


class ChatState(TypedDict):
    task: str
    age: Any
    history: List[Dict]
    output: str
    story_state: StoryTellerState
    next_node: str
    node_name: str


# ======================= Initialization ==========================
import random


def _initialize_state(Input) -> ChatState:
    age = random.randint(1, 80)
    return {
        "task": Input,
        "age": age,
        "history": [],
        "next_node": "",
        "node_name": "",
        "story_state": {
            "task": "",
            "age": age,
            "output": "",
            "retrieved_content": [],
            "history": [],
            "next_node": "",
            "node_name": "",
        },
    }
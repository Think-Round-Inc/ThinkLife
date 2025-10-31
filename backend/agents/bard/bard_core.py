import logging
import os
import uuid
import asyncio
from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite

from brain.brain_core import ThinkxLifeBrain
from agents.bard.tools import search_web
from agents.bard import states, prompts
from agents import utils


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


brain = ThinkxLifeBrain()

load_dotenv()
if brain.config["providers"]["gemini"]["enabled"]:
    from langchain_google_genai import ChatGoogleGenerativeAI

    LLM = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL"),
        temperature=0.9,  # only for the creative poets
        max_tokens=float(os.getenv("GEMINI_MAX_TOKENS")),
        api_key=os.getenv("GEMINI_API_KEY"),
    )
elif brain.config["providers"]["openai"]["enabled"]:
    from langchain_openai import ChatOpenAI

    LLM = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        temperature=os.getenv("OPENAI_TEMPERATURE"),
        max_tokens=os.getenv("OPENAI_MAX_TOKENS"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ===================== Story Teller Agent ==============================
class StoryTeller:
    """
    Raises awareness with entertaining, rhymey and short stories.
    have the right tone to narrate to kids and adults.
    searches the web for topics to make a story about.
    """

    def __init__(self, llm):
        self.llm = llm
        self.web_search_function = search_web

        build_story = StateGraph(states.StoryTellerState)

        build_story.add_node("search", self.search_node)
        build_story.add_node("adults", self.adults_node)
        build_story.add_node("kids", self.kids_node)

        build_story.set_entry_point("search")

        build_story.add_conditional_edges(
            "search", self.decision, {"adults": "adults", "kids": "kids"}
        )

        build_story.add_edge("adults", END)
        build_story.add_edge("kids", END)

        self.storyteller = build_story.compile()

    async def search_node(self, state: states.StoryTellerState):
        messages = [
            SystemMessage(content=prompts.SEARCH_PROMPT),
            HumanMessage(content=state.get("task", "")),
        ]
        search_queries = await self.llm.with_structured_output(states.Query).ainvoke(
            messages
        )

        search_results = []

        tasks = [
            self.web_search_function.ainvoke(
                {"query": q, "max_results": search_queries.max_results}
            )
            for q in search_queries.query
        ]
        responses = await asyncio.gather(*tasks)

        for response in responses:
            for item in response:
                search_results.append(
                    {
                        "url": item.get("url", ""),
                        "content": item.get("content", ""),
                    }
                )

        return {
            "node_name": "search",
            "retrieved_content": search_results,
            "task": state.get("task", ""),
        }

    async def adults_node(self, state: states.StoryTellerState):
        messages = [
            SystemMessage(content=prompts.STORYTELLER_PROMPT_ADULTS),
            HumanMessage(
                content=f"{state.get('task', '')} \n\n Here are the search results: \n\n {state.get('retrieved_content')}\n\n Here are the conversation history: \n\n {state.get('history')}"
            ),
        ]

        response = await self.llm.ainvoke(messages)

        return {
            "node_name": "adults",
            "task": state.get("task", ""),
            "output": response.content,
        }

    async def kids_node(self, state: states.StoryTellerState):
        messages = [
            SystemMessage(content=prompts.STORYTELLER_PROMPT_KIDS),
            HumanMessage(
                content=f"{state.get('task', '')} \n\n Here are the search results: \n\n {state.get('retrieved_content')}\n\n here are the conversation history: \n\n {state.get('history')}"
            ),
        ]

        response = await self.llm.ainvoke(messages)

        return {
            "node_name": "kids",
            "task": state.get("task", ""),
            "output": response.content,
        }

    def decision(self, state: states.StoryTellerState):
        if state.get("age", 21) < 16:
            return "kids"
        else:
            return "adults"


class Chatbot:
    """
    Main chatbot that utilizes the StoryTeller agent
    simple factory implementation, completely async agent
    """

    def __init__(self, llm, storyteller_agent, build_chatbot, conn, compiled_graph):
        self.llm = llm
        self.storyteller_agent = storyteller_agent
        self.build_chatbot = build_chatbot
        self.conn = conn
        self.chatbot = compiled_graph

    @classmethod
    async def build(cls):
        llm = LLM

        async def router_node(state: states.ChatState):
            messages = [
                SystemMessage(content=prompts.ROUTER_PROMPT),
                HumanMessage(
                    content=f"{state.get('task', '')} \n\n Retrieved content: \n\n {state.get('retrieved_content', '')}"
                ),
            ]

            response = await llm.with_structured_output(states.Router).ainvoke(messages)

            return {
                "task": state.get("task", ""),
                "next_node": response.next_node,
            }

        async def chat_node(state: states.ChatState):
            # Pull last 5 messages
            history = state.get("history", [])[-5:]

            messages = [
                SystemMessage(content=prompts.CHAT_PROMPT),
                *[
                    (
                        HumanMessage(content=m["content"])
                        if m["role"] == "user"
                        else AIMessage(content=m["content"])
                    )
                    for m in history
                ],
                HumanMessage(content=f'{state.get("task", "")}'),
            ]

            response = await llm.ainvoke(messages)

            # Update history
            new_history = state.get("history", [])
            new_history.append({"role": "user", "content": state["task"]})
            new_history.append({"role": "assistant", "content": response.content})

            return {
                "node_name": "chat",
                "task": state.get("task", ""),
                "output": response.content,
                "history": new_history,
            }

        async def storyteller_agent_node(state: states.ChatState):
            story_state = state["story_state"]
            story_state["task"] = state["task"]

            output = await storyteller_agent.ainvoke(story_state)

            new_history = state.get("history", [])
            new_history.append({"role": "user", "content": state["task"]})
            new_history.append(
                {"role": "assistant", "content": output.get("output", "")}
            )

            return {
                "node_name": "storyteller_agent",
                "story_state": output,
                "output": output.get("output", ""),
                "retrieved_content": output.get("retrieved_content", []),
                "history": new_history,
            }

        storyteller_agent = StoryTeller(llm).storyteller

        build_chatbot = StateGraph(states.ChatState)
        build_chatbot.add_node("router", router_node)
        build_chatbot.add_node("chat", chat_node)
        build_chatbot.add_node("storyteller", storyteller_agent_node)

        build_chatbot.add_edge("storyteller", END)
        build_chatbot.add_edge("chat", END)

        build_chatbot.set_entry_point("router")
        build_chatbot.add_conditional_edges(
            "router",
            lambda state: state.get("next_node", ""),
            {
                "chat": "chat",
                "storyteller": "storyteller",
            },
        )

        conn = await aiosqlite.connect("agents/checkpoints/checkpoints.sqlite")
        memory = AsyncSqliteSaver(conn)
        compile_kwargs = {"checkpointer": memory}

        compiled_graph = build_chatbot.compile(**compile_kwargs)

        return cls(llm, storyteller_agent, build_chatbot, conn, compiled_graph)

    async def close(self):
        if hasattr(self, "conn"):
            await self.conn.close()


class BardRunner:
    def __init__(self, chatbot: Chatbot, tracing: bool = False):
        self.bard = chatbot.chatbot
        self.thread_id = None
        self.config = {}
        self.tracing = tracing
        self.agent_name = "Bard"

        if tracing:
            utils.enable_tracing()

    async def new_thread(self, Input: str):
        self.thread_id = str(uuid.uuid4())
        self.config = {
            "configurable": {"thread_id": self.thread_id},
            "session_id": self.thread_id,
            "tags": [f"session:{self.thread_id}"], # for tracing
            "metadata": {"source": "BardRunner"},
        }
        state = states._initialize_state(Input)
        result = await self.bard.ainvoke(state, self.config)

        return {"session_id": self.thread_id, "output": result.get("output", "")}

    async def existing_thread(self, Input: str):
        if not self.thread_id:
            raise ValueError("No existing thread_id to resume")

        snapshot = await self.bard.aget_state(self.config)
        state = dict(snapshot.values)
        state["task"] = Input
        state["next_node"] = state.get("next_node", "chat")

        result = await self.bard.ainvoke(state, config=self.config)

        return {"session_id": self.thread_id, "output": result.get("output", "")}

    async def get_current_state(self, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        snapshot= await self.bard.aget_state(config)
        return snapshot.values
    def get_logs(self, limit: int = 20):
        """Fetches LLM logs for the entire session"""
        if not self.thread_id:
            raise ValueError("No existing thread_id to get logs")
        return utils.LLMLogs(self.thread_id, limit=limit)


if __name__ == "__main__":

    async def main():
        chatbot = await Chatbot.build()
        runner = BardRunner(chatbot, tracing=True)
        result = await runner.new_thread("tell me about Ancient Egypt legacy")
        print(result)

        logs = runner.get_logs()
        print(logs)

        await chatbot.close()

    asyncio.run(main())

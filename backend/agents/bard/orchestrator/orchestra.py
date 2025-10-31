import os
import asyncio
from dotenv import load_dotenv
from typing import Dict, Any, Literal
from pydantic import BaseModel
import time
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from brain.brain_core import ThinkxLifeBrain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from agents.bard import bard_core
except ImportError as e:
    logger.warning(f"Bard import failed: {e}")


# ======================== Initialize Brain and LLM ================================
brain = ThinkxLifeBrain()

load_dotenv()


def get_llm():
    # Handle both new BrainConfig object and legacy dict format
    providers = None
    if hasattr(brain.config, 'providers'):
        providers = brain.config.providers
    elif isinstance(brain.config, dict):
        providers = brain.config.get("providers", {})
    
    # Check for Gemini provider
    gemini_enabled = False
    if providers:
        if isinstance(providers, dict):
            gemini_config = providers.get("gemini", {})
            if isinstance(gemini_config, dict):
                gemini_enabled = gemini_config.get("enabled", False)
            else:
                gemini_enabled = getattr(gemini_config, 'enabled', False)
        else:
            gemini_provider = getattr(providers, 'gemini', None)
            if gemini_provider:
                gemini_enabled = getattr(gemini_provider, 'enabled', False)
    
    if gemini_enabled:
        from langchain_google_genai import ChatGoogleGenerativeAI

        LLM = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=0.9,  # only for the creative poets
            max_tokens=float(os.getenv("GEMINI_MAX_TOKENS", "2000")),
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    else:
        # Check for OpenAI provider
        openai_enabled = False
        if providers:
            if isinstance(providers, dict):
                openai_config = providers.get("openai", {})
                if isinstance(openai_config, dict):
                    openai_enabled = openai_config.get("enabled", False)
                else:
                    openai_enabled = getattr(openai_config, 'enabled', False)
            else:
                openai_provider = getattr(providers, 'openai', None)
                if openai_provider:
                    openai_enabled = getattr(openai_provider, 'enabled', False)
        if openai_enabled:
            from langchain_openai import ChatOpenAI

            LLM = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        else:
            # Fallback to OpenAI if no providers are configured
            from langchain_openai import ChatOpenAI
            LLM = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
                api_key=os.getenv("OPENAI_API_KEY"),
            )

    return LLM


class Router(BaseModel):
    next_agent: Literal["Bard"]  # we ONLY have Bard for now...


# TODO: add _handle_orchestrator to use this main orchestrator
# ================================= Build orchestrator ====================================
class Orchestrator:
    _instance = None

    # Singleton pattern
    def __new__(
        cls, llm, history: bool = True, tracing: bool = True, trace_limit: int = 20
    ):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.build(llm, tracing, history)
        return cls._instance

    def build(self, llm, tracing: bool, history: bool):
        self.llm = llm
        self.tracing = (
            tracing  # flag to GET LLM logs ( token usage, number of calls, ..etc)
        )
        self.history = history  # flag to GET the actual conversation history
        self.sessions = {}  # thread_id -> runner
        self.agents = {}

    async def _load_agent(self, agent_name: str):
        if agent_name in self.agents:
            return self.agents[agent_name]

        if agent_name == "Bard":
            chatbot = await bard_core.Chatbot.build()
            runner = bard_core.BardRunner(chatbot, tracing=self.tracing)

        ############ add elif for each future agent ( lazy loading ) ###########
        ############ make the runner the way you like, and use it ###########

        else:
            raise ValueError(f"Unknown agent: {agent_name}")

        self.agents[agent_name] = runner  # {{"Bard": BardRunner},{...:...}}
        return runner

    # the LLM decision is our routing condition for now
    async def llm_decision(self, Input: str):
        messages = [
            SystemMessage(
                content="""You are a decision maker. You are given a task and you decide which agent should handle it. 
                if the user asked for poetry route to Bard.
                """
            ),
            HumanMessage(content=Input),
        ]

        response = await self.llm.with_structured_output(Router).ainvoke(messages)

        return {"next_agent": response.next_agent}

    async def orchestrate(
        self,
        Input: str,
        agent_name: str = None,
        session_id: str = None,
        # tracing: bool = False,
        trace_limit: int = 20,
    ):

        start_time = time.time()

        if session_id is None:
            if not agent_name:
                decision = await self.llm_decision(Input)  # routing condition
                agent_name = decision["next_agent"]

            runner = await self._load_agent(agent_name)
            result = await runner.new_thread(Input)

            self.sessions[result["session_id"]] = runner
            session_id = result["session_id"]

        else:  # when routed to specific agent, the entire sessions is made inside that agent

            if session_id not in self.sessions:
                raise ValueError(f"Unknown session_id: {session_id}")
            runner = self.sessions[session_id]
            result = await runner.existing_thread(Input)
            agent_name = getattr(runner, "agent_name", None)

        if self.history:
            state = await runner.get_current_state(thread_id=session_id)
            result["history"] = state["history"]

        if self.tracing:
            result["trace"] = runner.get_logs(limit=trace_limit)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Orchestration with {agent_name} completed in {elapsed_time:.2f} seconds"
        )

        return {
            "session_id": session_id,
            "agent_name": agent_name,
            "output": result.get("output", "No output available"),
            "trace": result.get("trace", "No trace available"),
            "history": result.get("history", "No history available"),
        }

    async def close(self):
        """Gracefully close all agent resources and reset orchestrator state."""
        for agent_name, runner in self.agents.items():
            if hasattr(runner, "bard") and hasattr(runner.bard, "close"):
                await runner.bard.close()
        self.sessions.clear()
        self.agents.clear()
        Orchestrator._instance = None


# testing ( example usage )
if __name__ == "__main__":

    async def main():

        orchestrator = Orchestrator(
            llm=get_llm(), history=True, tracing=True, trace_limit=20
        )

        # Start a new session
        result = await orchestrator.orchestrate("write me a poem about the Nile")
        print("=== First Call ===")
        print(result)

        session_id = result["session_id"]

        # Continue same session
        followup = await orchestrator.orchestrate(
            "make it rhyme with pyramids", session_id=session_id
        )
        print("\n=== Follow Up ===")
        print(followup)

        await orchestrator.close()

    asyncio.run(main())

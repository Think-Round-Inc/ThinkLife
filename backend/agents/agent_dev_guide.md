### Agent Development Guide 

Hey team!
This system is built to be flexible and welcoming for anyone who wants to create their own agent. the idea is that you can build your agent the way you like, then plug it into the ecosystem with just a few simple steps.
Below will be some images for better illustration.
---
### Integration Steps

- Build your agent
Create your agent however you prefer, Keep it standalone at first, so you can test it independently.

- Define an entry point (Runner class)
Every agent should expose a Runner class as its main entry point.
All runners follow the same convention so that the Orchestrator can treat them equally (like a universal adapter). ( the runner convention is like BardRunner for example )

- Hook it into the Orchestrator
Inside the Orchestrator, add a simple elif statement to define how your agent is loaded.
This is the most important step: it makes the Orchestrator aware of your agent.

- Add prompt engineering and closing logic
In the llm_decision step, define routing rules so the system knows when to send tasks to your agent.
Update the Orchestrator’s close() function so it can gracefully shut down your agent when needed.
---
### Orchestrator overview

![overview](backend/agents/images/orchestrator_overview.png)
---
### How output looks like 

![example](backend/agents/images/orhcestrator_call.png)
"""
US 1.1 Task 3 - agent communication pattern: agents talk only through
AgentMessage envelopes delivered by the AgentRouter, never by sharing state.

Run with: pytest tests/test_agent_router.py -v
"""

from conftest import run
from brain.cortex.agent_router import AgentRouter
from brain.specs import AgentMessage, AgentResponse, MessageIntent, UserContext


class RecordingAgent:
    """Minimal IAgent stand-in that records what it receives."""

    def __init__(self, name, reply="ok", onward=None):
        self.name = name
        self.reply = reply
        self.onward = onward
        self.requests = []
        self.notifications = []

    async def process_request(self, request):
        self.requests.append(request)
        metadata = {"agent": self.name}
        if self.onward is not None:
            metadata["agent_message"] = self.onward
        return AgentResponse(success=True, content=self.reply, metadata=metadata)

    async def handle_notification(self, message):
        self.notifications.append(message)


USER = UserContext(user_id="u1", session_id="s1", is_authenticated=False)


def test_handoff_runs_recipient_with_only_the_envelope():
    router = AgentRouter()
    faq = RecordingAgent("faq", reply="We meet on Saturdays.")
    router.register("faq", faq)
    msg = AgentMessage(sender="zoe", recipient="faq", intent=MessageIntent.HANDOFF,
                       payload={"message": "When do you meet?"})

    response = run(router.dispatch(msg, USER))

    assert response.success and response.content == "We meet on Saturdays."
    assert response.metadata["correlation_id"] == msg.correlation_id
    assert response.metadata["handled_by"] == "faq"
    assert response.metadata["sent_by"] == "zoe"

    request = faq.requests[0]
    assert request.message == "When do you meet?"
    envelope = request.context.brain_context["agent_message"]
    assert envelope["sender"] == "zoe" and envelope["intent"] == "handoff"
    assert "payload" not in envelope  # nothing of the sender's beyond the message text
    assert request.context.conversation_history == []


def test_chained_handoff_keeps_one_correlation_id():
    router = AgentRouter()
    onward = AgentMessage(sender="faq", recipient="bard", intent="handoff", payload={"message": "Tell a story"})
    router.register("faq", RecordingAgent("faq", onward=onward))
    bard = RecordingAgent("bard", reply="Once upon a time")
    router.register("bard", bard)
    first = AgentMessage(sender="zoe", recipient="faq", intent="handoff", payload={"message": "story please"})

    response = run(router.dispatch(first, USER))

    assert response.content == "Once upon a time"
    assert response.metadata["correlation_id"] == first.correlation_id
    assert response.metadata["hop"] == 2


def test_hop_limit_stops_ping_pong():
    router = AgentRouter(max_hops=3)
    loop_back = AgentMessage(sender="b", recipient="a", intent="handoff", payload={"message": "back"})
    loop_fwd = AgentMessage(sender="a", recipient="b", intent="handoff", payload={"message": "fwd"})
    a, b = RecordingAgent("a", onward=loop_fwd), RecordingAgent("b", onward=loop_back)
    router.register("a", a)
    router.register("b", b)

    response = run(router.dispatch(AgentMessage(sender="user", recipient="a", intent="handoff",
                                                payload={"message": "start"}), USER))

    assert response.success is False
    assert "hop limit" in response.metadata["error"]
    assert len(a.requests) + len(b.requests) == 3


def test_unknown_recipient_is_refused_not_raised():
    response = run(AgentRouter().dispatch(
        AgentMessage(sender="zoe", recipient="bard", intent="handoff", payload={"message": "hi"}), USER))
    assert response.success is False
    assert "unknown recipient" in response.metadata["error"]


def test_handoff_without_message_text_is_refused():
    router = AgentRouter()
    router.register("faq", RecordingAgent("faq"))
    response = run(router.dispatch(AgentMessage(sender="zoe", recipient="faq", intent="query"), USER))
    assert response.success is False
    assert "payload['message']" in response.metadata["error"]


def test_notify_does_not_start_a_workflow():
    router = AgentRouter()
    faq = RecordingAgent("faq")
    router.register("FAQ", faq)  # names are case-insensitive
    msg = AgentMessage(sender="zoe", recipient="faq", intent=MessageIntent.NOTIFY, payload={"event": "session_ended"})

    response = run(router.dispatch(msg, USER))

    assert response.success is True
    assert faq.requests == []
    assert faq.notifications == [msg]
    assert router.registered() == ["faq"]

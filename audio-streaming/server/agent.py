# server/agent.py

from uuid import uuid4
from typing import AsyncIterator
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from .events import VoiceAgentEvent, AgentChunkEvent
import logging


logger = logging.getLogger(__name__)

# -------------------------
# Tools
# -------------------------

def add_to_order(item: str, quantity: int) -> str:
    """Add an item to the customer's order.
    
    Args:
        item: The name of the item (e.g., "pepperoni pizza")
        quantity: How many to add
    """
    return f"Added {quantity} x {item} to the order."

def confirm_order(order_summary: str) -> str:
    """Confirm the customer's final order and send it to the kitchen.
    
    Args:
        order_summary: Text summary of the complete order
    """
    return f"Order confirmed: {order_summary}. Sending to kitchen."


def cancel_order(order_to_cancel : str) -> str:
    """Cancel the already placed customers order
    
    Args:
        order_to_cancel : Get the order which the customer has placed and cancel it"""
    
    return f"Your order of {order_to_cancel} has been cancelled. Let me know if you want to order anything else."

tools = [add_to_order, confirm_order , cancel_order]

# -------------------------
# Agent Setup
# -------------------------

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
memory = InMemorySaver()

agent_executor = create_agent(
    model=model,
    tools=tools,
    checkpointer=memory,
)

SYSTEM_PROMPT = """
You are a friendly restaurant voice assistant.
Be concise, natural, and conversational.
Do NOT use emojis, markdown, or special characters.
Your responses will be spoken aloud by TTS.
Use tools when appropriate.
""".strip()

# -------------------------
# Session Management
# -------------------------

_session_threads: dict[str, str] = {}

def get_thread_id(session_id: str) -> str:
    if session_id not in _session_threads:
        _session_threads[session_id] = str(uuid4())
    return _session_threads[session_id]

def cleanup_thread(session_id: str):
    _session_threads.pop(session_id, None)

# -------------------------
# Agent Stream (Pipeline Stage)
# -------------------------

async def agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    session_id = "default"  # ✅ Will get from first event
    event_count = 0
    
    logger.info("Agent stream started")
    
    async for event in event_stream:
        event_count += 1
        
        # Get session_id from first event
        if hasattr(event, 'session_id') and event.session_id:
            session_id = event.session_id
        
        logger.debug(f"Agent received event {event_count}: {event.type}")
        
        # Pass through all upstream events
        yield event

        # Process final transcripts through agent
        if event.type == "stt_output":
            logger.info(f"Processing transcript: '{event.transcript}'")
            thread_id = get_thread_id(session_id)
            logger.debug(f"Using thread_id: {thread_id}")
            
            # Stream agent response
            stream = agent_executor.astream_events(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        HumanMessage(content=event.transcript)
                    ]
                },
                config={"configurable": {"thread_id": thread_id}},
                version="v2",
            )

            chunk_count = 0
            # Yield agent chunks as they arrive
            async for chunk_event in stream:
                if chunk_event["event"] == "on_chat_model_stream":
                    chunk = chunk_event["data"].get("chunk")
                    if chunk and getattr(chunk, "content", None):
                        chunk_count += 1
                        logger.debug(f"Agent chunk {chunk_count}: '{chunk.content}'")
                        yield AgentChunkEvent(text=chunk.content)
            
            logger.info(f"Agent completed, yielded {chunk_count} chunks")
    
    logger.info(f"Agent stream ended after {event_count} events")
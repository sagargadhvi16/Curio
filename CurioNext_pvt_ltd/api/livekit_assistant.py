# agent.py
import asyncio
import threading
from livekit.agents import Agent, AgentSession, JobContext
from livekit.plugins import groq, silero
import os

# Globals to manage agent state
_agent_thread = None
_agent_loop = None
_agent_ctx = None

def start_agent():
    global _agent_thread, _agent_loop, _agent_ctx

    def run():
        global _agent_loop, _agent_ctx
        _agent_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_agent_loop)
        _agent_ctx = JobContext()  # Fill in with required params if needed
        _agent_loop.run_until_complete(entrypoint(_agent_ctx))

    if _agent_thread and _agent_thread.is_alive():
        return False  # Already running
    _agent_thread = threading.Thread(target=run)
    _agent_thread.start()
    return True

def stop_agent():
    global _agent_loop, _agent_ctx, _agent_thread
    if _agent_ctx and _agent_loop:
        asyncio.run_coroutine_threadsafe(_agent_ctx.shutdown(reason="Disconnected by API"), _agent_loop)
        _agent_thread.join(timeout=5)
        _agent_ctx = None
        _agent_loop = None
        _agent_thread = None
        return True
    return False

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    agent = Agent(
        instructions="You are KidWatch, a friendly talking watch made for children ages 5-10 that can answer questions about anything."
    )
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=groq.STT(),
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        tts=groq.TTS(model="playai-tts"),
    )
    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="Cheerfully introduce yourself as Buddy. Mention you can answer questions about anything in a fun way. Keep it very brief and exciting for a child")
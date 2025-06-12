import asyncio
import threading
import os
from flask import Flask, jsonify
from flask_cors import CORS
from livekit.agents import Agent, AgentSession, JobContext
from livekit.plugins import groq, silero
from interest_analysis import analyze_past_week

# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local")

app = Flask(__name__)
CORS(app)

# Global state management
agent_state = {
    "thread": None,
    "loop": None,
    "job_ctx": None,
    "is_running": False,
    "session": None
}

def init_plugins():
    """Initialize required plugins and models"""
    # This would replace the 'download-files' command
    # Pre-load the VAD model to ensure it's downloaded before the server starts handling requests
    try:
        print("Initializing Silero VAD model...")
        # Just calling load() will trigger the download if needed
        vad = silero.VAD.load()
        print("VAD model initialized successfully")
        
        # Add any other initialization that was happening in the download-files command
        # For example, pre-loading TTS models, etc.
    except Exception as e:
        print(f"Error initializing plugins: {e}")
        # You might want to re-raise or exit if this is critical
        # raise

@app.route('/analyze', methods=['GET'])
def analyze():
    # Implement your analyze_past_week function
    result = analyze_past_week()
    return jsonify(result), 200

@app.route('/update_dashboard', methods=['GET'])
def update_dashboard():
    # Implement your update_dashboard function
    # This is a placeholder; replace with actual logic
    result = {"status": "success", "message": "Dashboard updated successfully"}
    return jsonify(result), 200


@app.route('/connect_assistant', methods=['POST'])
def connect_agent():
    global agent_state
    
    # Reset state if a previous session was incorrectly terminated
    if agent_state["is_running"] and (agent_state["thread"] is None or not agent_state["thread"].is_alive()):
        print("Detected stale agent state, resetting...")
        agent_state["is_running"] = False
        agent_state["loop"] = None
        agent_state["job_ctx"] = None
        agent_state["session"] = None
        
    if agent_state["is_running"]:
        print("Agent already running")
        return jsonify({"status": "already running", "message": "Agent is already connected"}), 200  # Changed to 200 to avoid frontend errors
    
    def run_agent():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            agent_state["loop"] = loop
            
            # Create required parameters for JobContext
            # Replace these with your actual LiveKit parameters
            job_ctx = JobContext(
                url=os.environ.get("LIVEKIT_URL", "ws://localhost:7880"),
                api_key=os.environ.get("LIVEKIT_API_KEY", "devkey"),
                api_secret=os.environ.get("LIVEKIT_API_SECRET", "secret"),
                room_name=os.environ.get("LIVEKIT_ROOM", "playground-czMi-7Ylv"),
                identity=os.environ.get("LIVEKIT_IDENTITY", "identity-5dqP")
            )

            
            print(f"Connecting to LiveKit room: {job_ctx.room_name} as {job_ctx.identity}")
            agent_state["job_ctx"] = job_ctx
            
            loop.run_until_complete(agent_entrypoint(job_ctx))
        except Exception as e:
            print(f"Agent error: {e}")
        finally:
            print("Agent thread finishing")
            agent_state["is_running"] = False
    
    try:
        print("Starting agent thread")
        agent_state["thread"] = threading.Thread(target=run_agent)
        agent_state["thread"].daemon = True  # Make thread daemon so it exits when main thread exits
        agent_state["thread"].start()
        agent_state["is_running"] = True
        
        return jsonify({"status": "connecting", "message": "Agent is connecting"}), 200
    except Exception as e:
        print(f"Failed to start agent thread: {e}")
        # Reset state on failure
        agent_state["is_running"] = False
        agent_state["thread"] = None
        return jsonify({"status": "error", "message": f"Failed to start agent: {str(e)}"}), 500

@app.route('/disconnect_assistant', methods=['POST'])
def disconnect_agent():
    global agent_state
    
    print("Disconnect request received")
    
    # If agent is not running according to our state
    if not agent_state["is_running"]:
        print("Agent not running - already disconnected")
        # Return success instead of error to make frontend happy
        return jsonify({"status": "not running", "message": "Agent was not running"}), 200
    
    try:
        if agent_state["loop"] and agent_state["job_ctx"]:
            print("Initiating agent shutdown")
            # Schedule the shutdown coroutine in the agent's event loop
            future = asyncio.run_coroutine_threadsafe(
                agent_state["job_ctx"].shutdown(reason="Disconnected by API"), 
                agent_state["loop"]
            )
            
            # Wait for the shutdown command to complete (with timeout)
            try:
                future.result(timeout=2)  # Short timeout for the command itself
                print("Shutdown command completed")
            except (asyncio.TimeoutError, Exception) as e:
                print(f"Shutdown command error (continuing anyway): {e}")
            
            # Wait for thread to finish (with timeout)
            if agent_state["thread"] and agent_state["thread"].is_alive():
                print("Waiting for agent thread to complete")
                agent_state["thread"].join(timeout=5)
                
                # If thread is still alive after timeout, we'll continue anyway
                if agent_state["thread"].is_alive():
                    print("WARNING: Agent thread did not shut down cleanly in time")
        else:
            print("No active loop or context found")
            
    except Exception as e:
        print(f"Error during disconnection: {e}")
    finally:
        # Always clean up state regardless of errors
        print("Resetting agent state")
        agent_state["loop"] = None
        agent_state["job_ctx"] = None
        agent_state["thread"] = None
        agent_state["is_running"] = False
        agent_state["session"] = None
        
    return jsonify({"status": "disconnected", "message": "Agent has been disconnected"}), 200

async def agent_entrypoint(ctx: JobContext):
    """The main entry point for the agent that runs in its own thread"""
    try:
        print("Starting agent entrypoint")
        # Connect to LiveKit
        print("Connecting to LiveKit...")
        await ctx.connect()
        print("Connected to LiveKit successfully")
        
        # Create the agent with instructions
        print("Creating agent...")
        agent = Agent(
            instructions="You are KidWatch, a friendly talking watch made for children ages 5-10 that can answer questions about anything."
        )
        
        # Create and start the session
        print("Creating session...")
        session = AgentSession(
            vad=silero.VAD.load(),
            stt=groq.STT(),
            llm=groq.LLM(model="llama-3.3-70b-versatile"),
            tts=groq.TTS(model="playai-tts"),
        )
        
        agent_state["session"] = session
        
        print("Starting agent session...")
        await session.start(agent=agent, room=ctx.room)
        print("Agent session started successfully")
        
        # Generate initial greeting
        print("Generating initial greeting...")
        await session.generate_reply(
            instructions="Cheerfully introduce yourself as Buddy. Mention you can answer questions about anything in a fun way. Keep it very brief and exciting for a child"
        )
        print("Initial greeting sent")
        
        # Keep the agent running until shutdown is called
        print("Agent ready and listening - waiting for shutdown signal")
        await ctx.wait_until_shutdown()
    except Exception as e:
        print(f"Error in agent entrypoint: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Agent entrypoint ending, cleaning up session")
        try:
            if agent_state["session"]:
                await agent_state["session"].stop()
                print("Agent session stopped cleanly")
        except Exception as e:
            print(f"Error stopping agent session: {e}")
        print("Agent has been shut down")

if __name__ == '__main__':
    # Initialize plugins before starting the server
    init_plugins()
    
    # Start the Flask server
    app.run(debug=True)
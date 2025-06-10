# main_api.py

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Response
from pydantic import BaseModel
import multiprocessing # Keeping for potential future use or if other parts still need it, but main assistant process removed
import time
import os
from typing import List, Dict, Any, Union
import uvicorn
import json
import whisper # Import whisper here to load model in API process
import torch # Import torch for device detection

# Import functions from your refactored voice assistant core
from voice_assistant_core import (
    process_incoming_audio_pipeline,
    WHISPER_MODEL_NAME,
    WHISPER_DEVICE,
    initialize_tts_client, # Added for direct TTS client initialization
    stt_model, # Accessing the global model instance (will be loaded here)
    CHAT_HISTORY_FILE # For chat history analysis
)
from interest_analysis import run_analysis

app = FastAPI(
    title="KidWatch Voice Assistant API",
    description="API to enable direct watch-to-cloud interaction for KidWatch."
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8080", "http://localhost:3001"], # Adjust as needed for your frontend/watch
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State for Models and TTS Client ---
# These will now be loaded directly in the FastAPI process
# We re-declare stt_model as a global here to ensure this process loads it.
# (The import 'from voice_assistant_core import stt_model' gets a reference,
# but we need to ensure it's loaded in *this* process context).
global stt_model # Declaring global again in api.py's scope

# This flag indicates if the core models (Whisper, Ollama connection) are ready
_core_models_ready: bool = False

class StatusResponse(BaseModel):
    status: str
    message: str

class ChatHistoryItem(BaseModel):
    role: str
    content: str
    timestamp: str

class ChatHistoryResponse(BaseModel):
    total_messages: int
    user_messages: int
    assistant_messages: int
    history: List[ChatHistoryItem]

@app.on_event("startup")
async def startup_event():
    """
    On application startup, load the Whisper STT model and initialize Google Cloud TTS client.
    """
    global stt_model, _core_models_ready

    print("FastAPI_Startup: Initializing KidWatch API services...")
    
    # 1. Initialize Whisper STT model
    current_whisper_device = WHISPER_DEVICE # Get the detected device from voice_assistant_core
    print(f"FastAPI_Startup: Whisper STT will run on: {current_whisper_device.upper()}")
    
    try:
        print(f"FastAPI_Startup: Whisper loading model '{WHISPER_MODEL_NAME}' on device '{current_whisper_device}'...")
        stt_model = whisper.load_model(WHISPER_MODEL_NAME, device=current_whisper_device)
        print(f"FastAPI_Startup: Whisper model '{WHISPER_MODEL_NAME}' loaded successfully on '{current_whisper_device}'.")
        if current_whisper_device == "cuda":
            print(f"FastAPI_Startup: Whisper FP16 will be used for transcription on CUDA device.")
    except Exception as e:
        print(f"FastAPI_Startup: CRITICAL - Error loading Whisper model on '{current_whisper_device}': {e}")
        print("FastAPI_Startup: Attempting Whisper fallback to CPU.")
        try:
            stt_model = whisper.load_model(WHISPER_MODEL_NAME, device="cpu")
            print(f"FastAPI_Startup: Whisper model '{WHISPER_MODEL_NAME}' loaded successfully on CPU (fallback).")
            # Update WHISPER_DEVICE in voice_assistant_core's scope if needed for consistency,
            # though the `process_incoming_audio_pipeline` will use the `stt_model` instance passed.
        except Exception as e_cpu:
            print(f"FastAPI_Startup: CRITICAL - Failed to load Whisper model on CPU as fallback: {e_cpu}")
            print("FastAPI_Startup: Assistant cannot start without STT model. API will be non-functional for speech.")
            stt_model = None # Ensure it's None if loading failed
            _core_models_ready = False
            return # Exit startup if model loading fails completely

    # 2. Initialize Google Cloud TTS client
    initialize_tts_client() # This function is in voice_assistant_core.py

    # 3. Ensure chat history file exists
    if not os.path.exists(CHAT_HISTORY_FILE):
        os.makedirs(os.path.dirname(CHAT_HISTORY_FILE) or '.', exist_ok=True)
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump({"items": []}, f) # Create an empty history if none exists
        print(f"FastAPI_Startup: Created empty chat history file at {CHAT_HISTORY_FILE}")

    # Mark core models as ready if Whisper loaded successfully
    if stt_model:
        _core_models_ready = True
        print("FastAPI_Startup: KidWatch API services are ready.")
    else:
        print("FastAPI_Startup: KidWatch API services NOT ready due to Whisper model loading failure.")


@app.on_event("shutdown")
async def shutdown_event():
    """
    On application shutdown, perform any necessary cleanup.
    """
    # No background process to stop anymore, but can add other cleanup if needed.
    print("FastAPI_Shutdown: Completed.")


@app.post("/assistant/process_watch_query", tags=["Watch Interaction"])
async def process_watch_query_endpoint(audio_file: UploadFile = File(...)):
    """
    Receives raw audio from the watch, transcribes it, generates an AI response,
    and returns the AI's spoken response audio directly.
    """
    global stt_model, _core_models_ready

    if not _core_models_ready or stt_model is None:
        raise HTTPException(status_code=503, detail="KidWatch API is not ready. STT model or TTS client not loaded.")

    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are accepted.")

    try:
        audio_bytes = await audio_file.read()
        print(f"API: Received audio file of size {len(audio_bytes)} bytes. Processing...")

        # Call the new central pipeline function from voice_assistant_core
        user_transcription, ai_response_text, ai_speech_audio_bytes = \
            process_incoming_audio_pipeline(audio_bytes, stt_model, WHISPER_DEVICE)

        if ai_speech_audio_bytes:
            print(f"API: Sending back AI response audio. Length: {len(ai_speech_audio_bytes)} bytes.")
            # Return the audio bytes directly as a response
            # Ensure the client is expecting 'audio/wav' (LINEAR16 encoding)
            return Response(content=ai_speech_audio_bytes, media_type="audio/wav")
        else:
            print("API Error: Failed to get AI speech audio bytes.")
            raise HTTPException(status_code=500, detail="Failed to generate AI speech audio response.")

    except HTTPException as e:
        # Re-raise explicit HTTP exceptions
        raise e
    except Exception as e:
        print(f"API: An unexpected error occurred in /assistant/process_watch_query: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


@app.get("/assistant/chat_analysis", tags=["Chat Analysis"])
async def get_chat_history_analysis():
    """
    Retrieves and provides a basic analysis of the chat history.
    """
    try:
        analysis_data = run_analysis()
        return analysis_data
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding chat history file. It might be corrupted.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/assistant/status", response_model=StatusResponse, tags=["API Status"])
async def get_api_status():
    """
    Checks the current operational status of the KidWatch API, including model loading.
    """
    global _core_models_ready
    if _core_models_ready:
        return {"status": "running", "message": "KidWatch API is ready and models are loaded."}
    else:
        return {"status": "starting_up", "message": "KidWatch API is starting up or encountered an error during model loading. Please check server logs."}

if __name__ == "__main__":
    # Ensure this runs the FastAPI app.
    print("Starting FastAPI server for KidWatch...")
    print("Make sure your Ollama server is running and accessible.")
    print("Make sure you have Google Cloud credentials configured (GOOGLE_APPLICATION_CREDENTIALS).")
    uvicorn.run(app, host="0.0.0.0", port=8000)
```




# main_api.py

# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from pydantic import BaseModel
# import multiprocessing
# import time
# import os
# from typing import List, Dict, Any
# import uvicorn
# import json

# # Import functions from your refactored voice assistant core
# from voice_assistant_core import run_assistant_process, CHAT_HISTORY_FILE
# from interest_analysis import run_analysis 
# from typing import Union
# app = FastAPI(
#     title="KidWatch Voice Assistant API",
#     description="API to control and get insights from the KidWatch voice assistant."
# )

# from fastapi.middleware.cors import CORSMiddleware

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://127.0.0.1:8000", "http://localhost:8080", "http://localhost:3001"],  # React default port
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# # --- Global State for Managing the Assistant Process ---
# # These need to be managed carefully as FastAPI can run multiple workers
# # For simplicity in this example, we assume a single worker or use a shared mechanism
# # In a production environment with multiple Uvicorn workers, you'd need a more robust
# # inter-process communication or a shared state manager (e.g., Redis) if the FastAPI
# # app itself scales across processes. For now, this works for single-worker Uvicorn.

# # assistant_process: multiprocessing.Process | None = None
# # stop_event: multiprocessing.Event | None = None # This event is passed to the child process

# assistant_process: Union[multiprocessing.Process, None] = None
# stop_event: Union[multiprocessing.Event, None] = None

# class StatusResponse(BaseModel):
#     status: str
#     message: str

# class ChatHistoryItem(BaseModel):
#     role: str
#     content: str
#     timestamp: str

# class ChatHistoryResponse(BaseModel):
#     total_messages: int
#     user_messages: int
#     assistant_messages: int
#     history: List[ChatHistoryItem]

# @app.on_event("startup")
# async def startup_event():
#     global assistant_process, stop_event
#     assistant_process = None
#     stop_event = None
#     print("FastAPI_Startup: Initialized global assistant state.")
#     # Clean up any stray history file if desired, or ensure permissions
#     if not os.path.exists(CHAT_HISTORY_FILE):
#         with open(CHAT_HISTORY_FILE, 'w') as f:
#             json.dump({"items": []}, f) # Create an empty history if none exists

# @app.on_event("shutdown")
# async def shutdown_event():
#     global assistant_process, stop_event
#     if assistant_process and assistant_process.is_alive():
#         print("FastAPI_Shutdown: Assistant process is alive. Attempting to stop it.")
#         if stop_event:
#             stop_event.set()
#         try:
#             assistant_process.join(timeout=10) # Wait for graceful shutdown
#             if assistant_process.is_alive():
#                 print("FastAPI_Shutdown: Assistant process did not terminate gracefully. Forcing termination.")
#                 assistant_process.terminate() # Force kill if it doesn't stop
#                 assistant_process.join() # Ensure it's cleaned up
#         except Exception as e:
#             print(f"FastAPI_Shutdown: Error during assistant process shutdown: {e}")
#     print("FastAPI_Shutdown: Completed.")


# @app.post("/assistant/start", response_model=StatusResponse, tags=["Assistant Control"])
# async def start_assistant_endpoint():
#     """
#     Starts the KidWatch voice assistant in a background process.
#     """
#     global assistant_process, stop_event
#     if assistant_process and assistant_process.is_alive():
#         raise HTTPException(status_code=409, detail="Assistant is already running.")

#     print("API: Received request to start assistant.")
#     stop_event = multiprocessing.Event()
#     # Use spawn context for better cross-platform compatibility if issues arise with fork
#     # ctx = multiprocessing.get_context('spawn')
#     # assistant_process = ctx.Process(target=run_assistant_process, args=(stop_event,))
#     assistant_process = multiprocessing.Process(target=run_assistant_process, args=(stop_event,))
#     assistant_process.daemon = True # So it exits if the main FastAPI app exits unexpectedly
    
#     try:
#         assistant_process.start()
#         # Give it a moment to initialize, especially model loading
#         time.sleep(1) # Small delay to check if it started or failed quickly
#         if not assistant_process.is_alive():
#              # This could happen if run_assistant_process exits very early due to an error
#              # (e.g., model loading failure, audio device issue)
#              # Check logs of voice_assistant_core.py for details.
#              assistant_process.join() # Clean up the dead process
#              assistant_process = None # Reset state
#              stop_event = None
#              raise HTTPException(status_code=500, detail="Assistant process failed to start or exited prematurely. Check server logs.")
#         print(f"API: Assistant process started with PID: {assistant_process.pid}")
#         return {"status": "success", "message": "KidWatch assistant started."}
#     except Exception as e:
#         print(f"API: Failed to start assistant process: {e}")
#         # Clean up if something went wrong during start
#         if assistant_process and assistant_process.is_alive():
#             assistant_process.terminate()
#             assistant_process.join()
#         assistant_process = None
#         stop_event = None
#         raise HTTPException(status_code=500, detail=f"Failed to start assistant process: {str(e)}")


# @app.post("/assistant/stop", response_model=StatusResponse, tags=["Assistant Control"])
# async def stop_assistant_endpoint():
#     """
#     Stops the KidWatch voice assistant.
#     """
#     global assistant_process, stop_event
#     if not assistant_process or not assistant_process.is_alive():
#         # Clear any stale references if process is dead but objects still exist
#         assistant_process = None
#         stop_event = None
#         raise HTTPException(status_code=404, detail="Assistant is not currently running.")

#     print("API: Received request to stop assistant.")
#     if stop_event:
#         stop_event.set() # Signal the process to stop
#     else:
#         # This case should ideally not happen if start was successful
#         print("API_Warning: stop_event is None but process is alive. Attempting terminate.")
#         assistant_process.terminate()
#         assistant_process.join(timeout=5)
#         if assistant_process.is_alive():
#              print("API_Warning: Process did not terminate after forceful attempt.")
#              return {"status": "error", "message": "Assistant running but could not be reliably stopped (no stop event). Force kill attempted."}
#         assistant_process = None
#         return {"status": "success", "message": "Assistant process likely terminated (no stop event)."}


#     try:
#         # Wait for the process to finish. The timeout is important.
#         assistant_process.join(timeout=15) # Adjust timeout as needed

#         if assistant_process.is_alive():
#             print("API: Assistant process did not stop gracefully within timeout. Terminating.")
#             assistant_process.terminate() # Force kill if it doesn't stop
#             assistant_process.join(timeout=5) # Wait for termination
#             message = "Assistant process did not stop gracefully and was terminated."
#         else:
#             print("API: Assistant process stopped gracefully.")
#             message = "KidWatch assistant stopped successfully."

#         assistant_process = None
#         stop_event = None
#         return {"status": "success", "message": message}

#     except Exception as e:
#         # This might happen if join() itself has an issue, though rare
#         print(f"API: Error during assistant stop: {e}")
#         # Attempt to clean up anyway
#         if assistant_process and assistant_process.is_alive():
#             assistant_process.terminate()
#             assistant_process.join(timeout=2)
#         assistant_process = None
#         stop_event = None
#         raise HTTPException(status_code=500, detail=f"Error stopping assistant: {str(e)}")


# @app.get("/assistant/chat_analysis", tags=["Chat anlysis"])
# async def get_chat_history_analysis():
#     """
#     Retrieves and provides a basic analysis of the chat history.
#     """

#     try:
#         analysis_data = run_analysis()
#         return analysis_data
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=500, detail="Error decoding chat history file. It might be corrupted.")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# @app.get("/assistant/status", response_model=StatusResponse, tags=["Assistant Control"])
# async def get_assistant_status():
#     """
#     Checks the current status of the KidWatch voice assistant.
#     """
#     global assistant_process
#     if assistant_process and assistant_process.is_alive():
#         return {"status": "running", "message": f"KidWatch assistant is running (PID: {assistant_process.pid})."}
#     else:
#         # Check if the process object exists but is dead (e.g. crashed)
#         if assistant_process and not assistant_process.is_alive():
#             # Optionally try to join it to clean up resources if not already done
#             # assistant_process.join(timeout=0.1) # Non-blocking join
#             # assistant_process = None # Clear the dead process reference
#             # stop_event = None
#             return {"status": "stopped", "message": "KidWatch assistant was running but has stopped (or crashed)."}
#         return {"status": "stopped", "message": "KidWatch assistant is not running."}

# if __name__ == "__main__":

#     # Important: Set multiprocessing start method to 'spawn' if you face issues on macOS or Windows
#     # This should be done *before* any multiprocessing objects are created.
#     # multiprocessing.set_start_method("spawn", force=True)
#     # Note: setting it here in __main__ might be too late if `app` object creation
#     # somehow involves multiprocessing indirectly earlier. Best place is top of the script.
    
#     print("Starting FastAPI server for KidWatch...")
#     print("Make sure your Ollama server is running.")
#     print("Make sure you have a microphone connected and configured.")
#     uvicorn.run(app, host="0.0.0.0", port=8000)

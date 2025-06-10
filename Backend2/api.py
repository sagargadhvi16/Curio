# main_api.py

from fastapi import FastAPI, HTTPException, File, UploadFile, Response
from pydantic import BaseModel
import os
import json
from typing import List, Dict, Any, Union

import uvicorn

# Import functions and global states from your refactored voice assistant core
from voice_assistant_core import (
    initialize_gcp_clients,
    process_incoming_audio_pipeline,
    CHAT_HISTORY_FILE,
    speech_client, # Access the global client objects to check their status
    tts_client,
    gemini_model
)
from interest_analysis import run_analysis # Keep this for chat analysis dashboard

app = FastAPI(
    title="KidWatch GCP Voice Assistant API",
    description="API to enable direct watch-to-cloud interaction for KidWatch, powered by Google Cloud services."
)

from fastapi.middleware.cors import CORSMiddleware

# Configure CORS to allow requests from your frontend and watch apps.
# IMPORTANT: Adjust 'allow_origins' for your specific deployment environment.
# For production, replace '*' with your actual domain(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8080", # Common for local React/JS development
        "http://localhost:3000", # Another common React port
        "http://localhost:3001"
        # Add your mobile/watch app's expected origin if applicable,
        # though for direct watch-to-cloud, it will be a direct API call.
        # When deployed, use the actual domain/IP of your frontend if it makes direct calls.
    ],
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods for simplicity in POC
    allow_headers=["*"], # Allow all headers
)

# --- Global State for Model Readiness ---
# This flag indicates if the core GCP services (Speech, Gemini, TTS) are ready
_gcp_services_ready: bool = False

# --- GCP Project and Location Configuration ---
# It's best practice to get these from environment variables in a deployed environment.
# If running locally without these set, provide sensible defaults for testing (e.g., your project ID and a region).
# For Vertex AI (Gemini), location is crucial (e.g., "us-central1").
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id") # REPLACE with your actual GCP Project ID
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1") # REPLACE with your preferred Vertex AI region (e.g., "us-central1")


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
    On application startup, initialize all Google Cloud clients (Speech, TTS, Gemini).
    """
    global _gcp_services_ready

    print("FastAPI_Startup: Initializing KidWatch API services with GCP...")
    print(f"FastAPI_Startup: Using GCP Project ID: {GCP_PROJECT_ID}, Location: {GCP_LOCATION}")
    
    # 1. Initialize all GCP clients
    initialize_gcp_clients(project_id=GCP_PROJECT_ID, location=GCP_LOCATION)

    # 2. Check if all clients were successfully initialized
    if speech_client is not None and tts_client is not None and gemini_model is not None:
        _gcp_services_ready = True
        print("FastAPI_Startup: All KidWatch GCP services (Speech, Gemini, TTS) are ready.")
    else:
        _gcp_services_ready = False
        print("FastAPI_Startup: KidWatch GCP services NOT fully ready. Check logs for initialization errors.")
        print("Hint: Ensure GOOGLE_APPLICATION_CREDENTIALS is set, and APIs are enabled for your project.")

    # 3. Ensure chat history file exists
    if not os.path.exists(CHAT_HISTORY_FILE):
        os.makedirs(os.path.dirname(CHAT_HISTORY_FILE) or '.', exist_ok=True)
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump({"items": []}, f) # Create an empty history if none exists
        print(f"FastAPI_Startup: Created empty chat history file at {CHAT_HISTORY_FILE}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    On application shutdown, perform any necessary cleanup.
    (No long-running processes to terminate in this GCP-native version).
    """
    print("FastAPI_Shutdown: Completed.")


@app.post("/assistant/process_watch_query", tags=["Watch Interaction"])
async def process_watch_query_endpoint(audio_file: UploadFile = File(...)):
    """
    Receives raw audio from the watch, transcribes it using GCP STT,
    generates an AI response using Gemini, and returns the AI's spoken response audio (GCP TTS).
    """
    global _gcp_services_ready

    if not _gcp_services_ready:
        raise HTTPException(status_code=503, detail="KidWatch API is not ready. GCP services not initialized.")

    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are accepted.")

    try:
        audio_bytes = await audio_file.read()
        print(f"API: Received audio file of size {len(audio_bytes)} bytes. Processing with GCP...")

        # Call the central pipeline function from voice_assistant_core, passing GCP details
        user_transcription, ai_text_response, ai_speech_audio_bytes = \
            process_incoming_audio_pipeline(audio_bytes, project_id=GCP_PROJECT_ID, location=GCP_LOCATION)

        if ai_speech_audio_bytes:
            print(f"API: Sending back AI response audio. Length: {len(ai_speech_audio_bytes)} bytes.")
            # Return the audio bytes directly as a response
            # Ensure the client is expecting 'audio/wav' (LINEAR16 encoding from TTS)
            return Response(content=ai_speech_audio_bytes, media_type="audio/wav")
        else:
            print("API Error: Failed to get AI speech audio bytes from pipeline.")
            raise HTTPException(status_code=500, detail="Failed to generate AI speech audio response.")

    except HTTPException as e:
        # Re-raise explicit HTTP exceptions
        raise e
    except Exception as e:
        print(f"API: An unexpected error occurred in /assistant/process_watch_query: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
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
        print(f"API: Error in chat analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during chat analysis: {str(e)}")

@app.get("/assistant/status", response_model=StatusResponse, tags=["API Status"])
async def get_api_status():
    """
    Checks the current operational status of the KidWatch API, including GCP service initialization.
    """
    global _gcp_services_ready
    if _gcp_services_ready:
        return {"status": "running", "message": "KidWatch API is ready and GCP services are initialized."}
    else:
        return {"status": "starting_up", "message": "KidWatch API is starting up or encountered an error during GCP service initialization. Please check server logs."}


if __name__ == "__main__":
    # --- IMPORTANT LOCAL RUNNING INSTRUCTIONS ---
    print("--- Starting FastAPI server for KidWatch (GCP Backend) ---")
    print("BEFORE RUNNING:")
    print("1. Ensure you have activated a virtual environment and installed dependencies:")
    print("   pip install -r Backend2/requirements.txt")
    print("2. Set your Google Cloud Project ID and Location as environment variables:")
    print(f"   export GCP_PROJECT_ID=\"your-gcp-project-id\"") # Replace with your actual project ID
    print(f"   export GCP_LOCATION=\"us-central1\"") # Replace with your Vertex AI region (e.g., us-central1)
    print("3. Set your Google Cloud service account key path:")
    print("   export GOOGLE_APPLICATION_CREDENTIALS=\"/path/to/your/service-account-key.json\"")
    print("   (This JSON file should be downloaded from GCP Console, IAM & Admin -> Service Accounts)")
    print("4. Ensure billing is enabled for your GCP project and relevant APIs are enabled:")
    print("   - Cloud Speech-to-Text API")
    print("   - Cloud Text-to-Speech API")
    print("   - Vertex AI API")
    print("----------------------------------------------------------")

    # The actual FastAPI server run command
    uvicorn.run(app, host="0.0.0.0", port=8000)

```

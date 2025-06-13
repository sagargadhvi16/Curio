# api.py

from fastapi import FastAPI, HTTPException, File, UploadFile, Response
from contextlib import asynccontextmanager
from pydantic import BaseModel
import os
import json
from typing import List, Dict, Any, Union
import uvicorn

# Import functions and global states using relative imports,
# as this is part of the 'Backend2' Python package.
from .voice_assistant_core import (
    initialize_gcp_clients,
    process_incoming_audio_pipeline,
    CHAT_HISTORY_FILE,
    speech_client,
    tts_client,
    gemini_model
)
from .interest_analysis import run_analysis

# --- Global State for Model Readiness ---
_gcp_services_ready: bool = False

# --- GCP Project and Location Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global _gcp_services_ready
    print("FastAPI_Startup: Initializing KidWatch API services with GCP...")
    print(f"FastAPI_Startup: Using GCP Project ID: {GCP_PROJECT_ID}, Location: {GCP_LOCATION}")
    
    try:
        initialize_gcp_clients(project_id=GCP_PROJECT_ID, location=GCP_LOCATION)

        # FIXED: Check if clients are actually initialized
        from .voice_assistant_core import speech_client, tts_client, gemini_model
        
        if speech_client is not None and tts_client is not None and gemini_model is not None:
            _gcp_services_ready = True
            print("FastAPI_Startup: All KidWatch GCP services (Speech, Gemini, TTS) are ready.")
        else:
            _gcp_services_ready = False
            print("FastAPI_Startup: KidWatch GCP services NOT fully ready. Check logs for initialization errors.")
            print("Hint: Ensure GOOGLE_APPLICATION_CREDENTIALS is set, and APIs are enabled for your project.")

    except Exception as e:
        _gcp_services_ready = False
        print(f"FastAPI_Startup: Error during GCP initialization: {e}")

    if not os.path.exists(CHAT_HISTORY_FILE):
        os.makedirs(os.path.dirname(CHAT_HISTORY_FILE) or '.', exist_ok=True)
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump({"items": []}, f)
        print(f"FastAPI_Startup: Created empty chat history file at {CHAT_HISTORY_FILE}")
    
    yield
    
    # Shutdown
    print("FastAPI_Shutdown: Completed.")
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup
#     global _gcp_services_ready
#     print("FastAPI_Startup: Initializing KidWatch API services with GCP...")
#     print(f"FastAPI_Startup: Using GCP Project ID: {GCP_PROJECT_ID}, Location: {GCP_LOCATION}")
    
#     initialize_gcp_clients(project_id=GCP_PROJECT_ID, location=GCP_LOCATION)

#     if speech_client is not None and tts_client is not None and gemini_model is not None:
#         _gcp_services_ready = True
#         print("FastAPI_Startup: All KidWatch GCP services (Speech, Gemini, TTS) are ready.")
#     else:
#         _gcp_services_ready = False
#         print("FastAPI_Startup: KidWatch GCP services NOT fully ready. Check logs for initialization errors.")
#         print("Hint: Ensure GOOGLE_APPLICATION_CREDENTIALS is set, and APIs are enabled for your project.")

#     if not os.path.exists(CHAT_HISTORY_FILE):
#         os.makedirs(os.path.dirname(CHAT_HISTORY_FILE) or '.', exist_ok=True)
#         with open(CHAT_HISTORY_FILE, 'w') as f:
#             json.dump({"items": []}, f)
#         print(f"FastAPI_Startup: Created empty chat history file at {CHAT_HISTORY_FILE}")
    
#     yield
    
#     # Shutdown
#     print("FastAPI_Shutdown: Completed.")

app = FastAPI(
    title="KidWatch GCP Voice Assistant API",
    description="API to enable direct watch-to-cloud interaction for KidWatch, powered by Google Cloud services.",
    lifespan=lifespan
)

from fastapi.middleware.cors import CORSMiddleware

# Configure CORS to allow requests from your frontend and watch apps.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8080",
        "http://localhost:3000",
        "http://localhost:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# Replace this in your Backend2/api.py

@app.post("/assistant/process_watch_query", tags=["Watch Interaction"])
async def process_watch_query_endpoint(audio_file: UploadFile = File(...)):
    global _gcp_services_ready
    if not _gcp_services_ready:
        raise HTTPException(status_code=503, detail="KidWatch API is not ready. GCP services not initialized.")
    
    # More flexible content type checking
    valid_content_types = [
        "audio/wav", "audio/wave", "audio/x-wav",
        "audio/mpeg", "audio/mp3", 
        "audio/mp4", "audio/m4a",
        "audio/flac", "audio/x-flac",
        "audio/ogg", "application/ogg",
        "audio/webm",
        "audio/amr",
        "application/octet-stream",  # For cases where type isn't detected
        None  # For cases where content_type is None
    ]
    
    # Check file extension as fallback
    filename_lower = audio_file.filename.lower() if audio_file.filename else ""
    valid_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm', '.amr']
    
    # Validate either by content type OR file extension
    content_type_valid = (audio_file.content_type is None or 
                         audio_file.content_type in valid_content_types or
                         (audio_file.content_type and audio_file.content_type.startswith("audio/")))
    
    extension_valid = any(filename_lower.endswith(ext) for ext in valid_extensions)
    
    if not (content_type_valid or extension_valid):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Received content-type: {audio_file.content_type}, filename: {audio_file.filename}. Only audio files are accepted."
        )
    
    try:
        audio_bytes = await audio_file.read()
        print(f"API: Received audio file '{audio_file.filename}' (content-type: {audio_file.content_type}) of size {len(audio_bytes)} bytes. Processing with GCP...")
        
        user_transcription, ai_text_response, ai_speech_audio_bytes = \
            process_incoming_audio_pipeline(audio_bytes, project_id=GCP_PROJECT_ID, location=GCP_LOCATION)
            
        if ai_speech_audio_bytes:
            print(f"API: Sending back AI response audio. Length: {len(ai_speech_audio_bytes)} bytes.")
            return Response(content=ai_speech_audio_bytes, media_type="audio/wav")
        else:
            print("API Error: Failed to get AI speech audio bytes from pipeline.")
            raise HTTPException(status_code=500, detail="Failed to generate AI speech audio response.")
            
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"API: An unexpected error occurred in /assistant/process_watch_query: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
# @app.post("/assistant/process_watch_query", tags=["Watch Interaction"])
# async def process_watch_query_endpoint(audio_file: UploadFile = File(...)):
#     global _gcp_services_ready
#     if not _gcp_services_ready:
#         raise HTTPException(status_code=503, detail="KidWatch API is not ready. GCP services not initialized.")

#     if not audio_file.content_type.startswith("audio/"):
#         raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are accepted.")

#     try:
#         audio_bytes = await audio_file.read()
#         print(f"API: Received audio file of size {len(audio_bytes)} bytes. Processing with GCP...")

#         user_transcription, ai_text_response, ai_speech_audio_bytes = \
#             process_incoming_audio_pipeline(audio_bytes, project_id=GCP_PROJECT_ID, location=GCP_LOCATION)

#         if ai_speech_audio_bytes:
#             print(f"API: Sending back AI response audio. Length: {len(ai_speech_audio_bytes)} bytes.")
#             return Response(content=ai_speech_audio_bytes, media_type="audio/wav")
#         else:
#             print("API Error: Failed to get AI speech audio bytes from pipeline.")
#             raise HTTPException(status_code=500, detail="Failed to generate AI speech audio response.")
            
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         print(f"API: An unexpected error occurred in /assistant/process_watch_query: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/assistant/chat_analysis", tags=["Chat Analysis"])
async def get_chat_history_analysis():
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
    global _gcp_services_ready
    if _gcp_services_ready:
        return {"status": "running", "message": "KidWatch API is ready and GCP services are initialized."}
    else:
        return {"status": "starting_up", "message": "KidWatch API is starting up or encountered an error during GCP service initialization. Please check server logs."}

@app.get("/")
async def root():
    return {"message": "KidWatch API is running", "status": "healthy"}

if __name__ == "__main__":
    print("--- Starting FastAPI server for KidWatch (GCP Backend) ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)

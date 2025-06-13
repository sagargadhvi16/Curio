# voice_assistant_core.py

import os
import json
from datetime import datetime
import traceback
from typing import Tuple, List, Dict, Optional, Any

# --- Import Google Cloud Libraries ---
try:
    from google.cloud import speech, texttospeech, aiplatform
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
except ImportError:
    print("Error: Required Google Cloud libraries not found.")
    print("Please install them using: pip install google-cloud-speech google-cloud-texttospeech google-cloud-aiplatform")
    speech = texttospeech = aiplatform = vertexai = GenerativeModel = Part = None


# --- Global State Management ---
speech_client: Optional[speech.SpeechClient] = None
tts_client: Optional[texttospeech.TextToSpeechClient] = None
gemini_model: Optional[GenerativeModel] = None

CHAT_HISTORY_FILE = os.path.join("Backend2", "data", "chat_history.json")

# Update this in Backend2/voice_assistant_core.py

def initialize_gcp_clients(project_id: str, location: str):
    """
    Initializes all necessary Google Cloud clients and the Vertex AI SDK.
    """
    global speech_client, tts_client, gemini_model
    
    if speech_client and tts_client and gemini_model:
        print("Core: GCP clients are already initialized.")
        return

    print(f"Core: Initializing GCP clients for project '{project_id}' in location '{location}'...")
    try:
        speech_client = speech.SpeechClient()
        print("Core: Google Cloud Speech-to-Text client initialized successfully.")

        tts_client = texttospeech.TextToSpeechClient()
        print("Core: Google Cloud Text-to-Speech client initialized successfully.")

        vertexai.init(project=project_id, location=location)
        print(f"Core: Vertex AI SDK initialized for project '{project_id}' and location '{location}'.")

        # Try different Gemini models (in order of preference)
        model_names = [
            "gemini-1.5-flash",      # Latest and most available
            "gemini-1.5-pro",        # Alternative latest
            "gemini-pro",            # Simple name
            "gemini-1.0-pro",        # Without the -001 suffix
        ]
        
        gemini_model = None
        for model_name in model_names:
            try:
                gemini_model = GenerativeModel(model_name)
                print(f"Core: Successfully loaded Gemini model: {model_name}")
                break
            except Exception as e:
                print(f"Core: Failed to load {model_name}: {e}")
                continue
        
        if not gemini_model:
            raise Exception("No Gemini model could be loaded")

    except Exception as e:
        print(f"Core_Error: Failed to initialize one or more GCP services.")
        print(f"Core_Error: Details - {e}")
        speech_client = tts_client = gemini_model = None

# # 2. voice_assistant_core.py - Fixed initialization function

# def initialize_gcp_clients(project_id: str, location: str):
#     """
#     Initializes all necessary Google Cloud clients and the Vertex AI SDK.
#     """
#     global speech_client, tts_client, gemini_model
    
#     if speech_client and tts_client and gemini_model:
#         print("Core: GCP clients are already initialized.")
#         return

#     print(f"Core: Initializing GCP clients for project '{project_id}' in location '{location}'...")
#     try:
#         # Initialize Speech-to-Text client
#         speech_client = speech.SpeechClient()
#         print("Core: Google Cloud Speech-to-Text client initialized successfully.")

#         # Initialize Text-to-Speech client
#         tts_client = texttospeech.TextToSpeechClient()
#         print("Core: Google Cloud Text-to-Speech client initialized successfully.")

#         # Initialize Vertex AI SDK
#         vertexai.init(project=project_id, location=location)
#         print(f"Core: Vertex AI SDK initialized for project '{project_id}' and location '{location}'.")

#         # Load the specific Gemini model
#         gemini_model = GenerativeModel("gemini-1.0-pro-001")
#         print("Core: Gemini Pro model loaded successfully.")

#         # IMPORTANT: Add a success confirmation
#         print("Core: ALL GCP services initialized successfully!")

#     except Exception as e:
#         print(f"Core_Error: Failed to initialize one or more GCP services.")
#         print(f"Core_Error: Details - {e}")
#         print("Core_Hint: Ensure your GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly.")
#         print("Core_Hint: Make sure you have enabled required APIs in your GCP project.")
#         speech_client = tts_client = gemini_model = None
#         raise e  # Re-raise to be caught by the startup handler
        
# 1. def initialize_gcp_clients(project_id: str, location: str):
#     """
#     Initializes all necessary Google Cloud clients and the Vertex AI SDK.
#     """
#     # --- THE FIX IS HERE ---
#     # The 'global' keyword tells this function to modify the global variables
#     # defined at the top of the file, instead of creating new local ones.
#     global speech_client, tts_client, gemini_model
    
#     if speech_client and tts_client and gemini_model:
#         print("Core: GCP clients are already initialized.")
#         return

#     print(f"Core: Initializing GCP clients for project '{project_id}' in location '{location}'...")
#     try:
#         speech_client = speech.SpeechClient()
#         print("Core: Google Cloud Speech-to-Text client initialized successfully.")

#         tts_client = texttospeech.TextToSpeechClient()
#         print("Core: Google Cloud Text-to-Speech client initialized successfully.")

#         vertexai.init(project=project_id, location=location)
#         print(f"Core: Vertex AI SDK initialized for project '{project_id}' and location '{location}'.")

#         gemini_model = GenerativeModel("gemini-1.0-pro-001")
#         print("Core: Gemini Pro model loaded successfully.")

#     except Exception as e:
#         print(f"Core_Error: Failed to initialize one or more GCP services.")
#         print(f"Core_Error: Details - {e}")
#         print("Core_Hint: Ensure your GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly.")
#         print("Core_Hint: Make sure you have enabled 'Cloud Speech-to-Text API', 'Cloud Text-to-Speech API', and 'Vertex AI API' in your GCP project.")
#         speech_client = tts_client = gemini_model = None


def load_chat_history() -> List[Dict[str, Any]]:
    """
    Loads the chat history from the JSON file.
    """
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r') as f:
                data = json.load(f)
                return data.get("items", [])
        return []
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Core_Warning: Could not load chat history from {CHAT_HISTORY_FILE}. Starting fresh. Error: {e}")
        return []

def save_chat_history_to_file(history: List[Dict[str, Any]]):
    """
    Saves the updated chat history back to the JSON file.
    """
    try:
        os.makedirs(os.path.dirname(CHAT_HISTORY_FILE) or '.', exist_ok=True)
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump({"items": history}, f, indent=4)
    except Exception as e:
        print(f"Core_Error: Failed to save chat history to {CHAT_HISTORY_FILE}. Error: {e}")

# Replace the transcribe_audio_gcp function in Backend2/voice_assistant_core.py

def transcribe_audio_gcp(audio_bytes: bytes) -> Optional[str]:
    """
    Transcribes the given audio bytes to text using Google Cloud Speech-to-Text.
    Auto-detects sample rate from WAV header.
    """
    if not speech_client:
        print("Core_Error: Speech-to-Text client is not initialized.")
        return None

    try:
        # Prepare the audio
        audio = speech.RecognitionAudio(content=audio_bytes)
        
        # Use auto-detection for sample rate and encoding
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            # Remove sample_rate_hertz to let Google auto-detect
            language_code="en-US",
            model="latest_long",
            enable_automatic_punctuation=True,  # Better transcription
            use_enhanced=True,  # Use enhanced model if available
        )

        print("Core: Sending audio to GCP for transcription (auto-detecting sample rate)...")
        response = speech_client.recognize(config=config, audio=audio)
        print("Core: Received transcription response from GCP.")

        if response.results and response.results[0].alternatives:
            transcript = response.results[0].alternatives[0].transcript.strip()
            confidence = response.results[0].alternatives[0].confidence
            print(f"Core: Transcription successful. Text: '{transcript}' (confidence: {confidence:.2f})")
            
            if len(transcript) < 1:
                print("Core_Warning: Transcription is empty or too short.")
                return None
                
            return transcript
        else:
            print("Core_Warning: Transcription returned no results. The audio might be silent or unclear.")
            return None

    except Exception as e:
        print(f"Core_Error: An error occurred during GCP transcription: {e}")
        traceback.print_exc()
        return None

# def transcribe_audio_gcp(audio_bytes: bytes) -> Optional[str]:
#     """
#     Transcribes the given audio bytes to text using Google Cloud Speech-to-Text.
#     """
#     if not speech_client:
#         print("Core_Error: Speech-to-Text client is not initialized.")
#         return None

#     try:
#         audio = speech.RecognitionAudio(content=audio_bytes)
#         config = speech.RecognitionConfig(
#             encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#             sample_rate_hertz=16000,
#             language_code="en-US",
#             model="latest_long",
#         )

#         print("Core: Sending audio to GCP for transcription...")
#         response = speech_client.recognize(config=config, audio=audio)
#         print("Core: Received transcription response from GCP.")

#         if response.results and response.results[0].alternatives:
#             transcript = response.results[0].alternatives[0].transcript
#             print(f"Core: Transcription successful. Text: '{transcript}'")
#             return transcript
#         else:
#             print("Core_Warning: Transcription returned no results. The audio might be silent or unclear.")
#             return None

#     except Exception as e:
#         print(f"Core_Error: An error occurred during GCP transcription: {e}")
#         traceback.print_exc()
#         return None


def get_gemini_response(user_input: str, project_id: str, location: str) -> Optional[str]:
    """
    Generates a text response from Gemini based on the user's input and chat history.
    """
    if not gemini_model:
        print("Core_Warning: Gemini model not initialized. Attempting re-initialization...")
        initialize_gcp_clients(project_id, location)
        if not gemini_model:
            print("Core_Error: Gemini model could not be initialized. Cannot generate response.")
            return None

    try:
        chat_history = load_chat_history()
        gemini_history = []
        for message in chat_history:
            role = "model" if message["role"] == "assistant" else "user"
            gemini_history.append(Part.from_text(message["content"], role=role))
        
        chat = gemini_model.start_chat(history=gemini_history)
        
        persona_prompt = (
            "You are Curio, a friendly, curious, and very patient AI companion for a young child. "
            "Your purpose is to answer questions in a simple, encouraging, and easy-to-understand way. "
            "Always be positive and supportive. Keep your answers concise, ideally 1-3 sentences. "
            "If you don't know an answer, say so in a simple way, like 'That's a great question! I'm not sure about that one, but we can learn together.' "
            "Never use complex words or jargon. Your goal is to spark curiosity, not to be a walking encyclopedia."
        )

        full_prompt = f"{persona_prompt}\n\nChild asks: \"{user_input}\""
        
        print("Core: Sending prompt to Gemini...")
        response = chat.send_message(full_prompt)
        print("Core: Received response from Gemini.")

        ai_response_text = response.text
        print(f"Core: Gemini response text: '{ai_response_text}'")

        timestamp = datetime.now().isoformat()
        chat_history.append({"role": "user", "content": user_input, "timestamp": timestamp})
        chat_history.append({"role": "assistant", "content": ai_response_text, "timestamp": timestamp})

        save_chat_history_to_file(chat_history)
        print("Core: Chat history has been updated and saved.")
        
        return ai_response_text

    except Exception as e:
        print(f"Core_Error: An error occurred while interacting with Gemini: {e}")
        traceback.print_exc()
        return None


def synthesize_speech_gcp(text_input: str) -> Optional[bytes]:
    """
    Converts text into speech audio bytes using Google Cloud Text-to-Speech.
    """
    if not tts_client:
        print("Core_Error: Text-to-Speech client is not initialized.")
        return None

    try:
        synthesis_input = texttospeech.SynthesisInput(text=text_input)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-G",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=0.95,
            pitch=-2.0
        )

        print("Core: Sending text to GCP for speech synthesis...")
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        print("Core: Received synthesized audio from GCP.")
        
        return response.audio_content

    except Exception as e:
        print(f"Core_Error: An error occurred during GCP speech synthesis: {e}")
        traceback.print_exc()
        return None


def process_incoming_audio_pipeline(
    audio_bytes: bytes,
    project_id: str,
    location: str
) -> Tuple[Optional[str], Optional[str], Optional[bytes]]:
    """
    The main pipeline that orchestrates the entire process.
    """
    print("\n--- Core: Starting KidWatch Audio Processing Pipeline ---")
    
    user_transcription = transcribe_audio_gcp(audio_bytes)
    if not user_transcription:
        print("Core_Pipeline_Error: Transcription failed. Aborting pipeline.")
        return None, None, None
    
    ai_text_response = get_gemini_response(user_transcription, project_id, location)
    if not ai_text_response:
        print("Core_Pipeline_Error: Failed to get response from Gemini. Aborting pipeline.")
        return user_transcription, None, None
        
    ai_speech_audio_bytes = synthesize_speech_gcp(ai_text_response)
    if not ai_speech_audio_bytes:
        print("Core_Pipeline_Error: Speech synthesis failed. Aborting pipeline.")
        return user_transcription, ai_text_response, None
    
    print("--- Core: KidWatch Audio Processing Pipeline Completed Successfully ---\n")
    return user_transcription, ai_text_response, ai_speech_audio_bytes


# # voice_assistant_core.py

# import os
# import json
# from datetime import datetime
# import traceback
# from typing import Tuple, List, Dict, Optional, Any

# # --- Import Google Cloud Libraries ---
# # Use try-except blocks to provide helpful error messages if libraries are not installed.
# try:
#     from google.cloud import speech, texttospeech, aiplatform
#     import vertexai
#     from vertexai.generative_models import GenerativeModel, Part
# except ImportError:
#     print("Error: Required Google Cloud libraries not found.")
#     print("Please install them using: pip install google-cloud-speech google-cloud-texttospeech google-cloud-aiplatform")
#     # Exit or handle appropriately if the script cannot run without these.
#     speech = texttospeech = aiplatform = vertexai = GenerativeModel = Part = None


# # --- Global State Management ---
# # Define global variables for the GCP clients and the Gemini model.
# # This prevents re-initialization on every API call, which is inefficient.
# speech_client: Optional[speech.SpeechClient] = None
# tts_client: Optional[texttospeech.TextToSpeechClient] = None
# gemini_model: Optional[GenerativeModel] = None

# # Define the path for the chat history file.
# # Using os.path.join ensures it works across different operating systems.
# # The 'data' directory will be created by the API if it doesn't exist.
# CHAT_HISTORY_FILE = os.path.join("Backend2", "data", "chat_history.json")


# def initialize_gcp_clients(project_id: str, location: str):
#     """
#     Initializes all necessary Google Cloud clients and the Vertex AI SDK.
#     - Speech-to-Text Client
#     - Text-to-Speech Client
#     - Vertex AI (for Gemini)

#     This function should be called once at application startup.

#     Args:
#         project_id (str): Your Google Cloud project ID.
#         location (str): The GCP region for Vertex AI services (e.g., "us-central1").
#     """
#     global speech_client, tts_client, gemini_model
    
#     # Check if clients are already initialized to avoid redundant calls.
#     if speech_client and tts_client and gemini_model:
#         print("Core: GCP clients are already initialized.")
#         return

#     print(f"Core: Initializing GCP clients for project '{project_id}' in location '{location}'...")
#     try:
#         # Initialize Speech-to-Text client
#         speech_client = speech.SpeechClient()
#         print("Core: Google Cloud Speech-to-Text client initialized successfully.")

#         # Initialize Text-to-Speech client
#         tts_client = texttospeech.TextToSpeechClient()
#         print("Core: Google Cloud Text-to-Speech client initialized successfully.")

#         # Initialize Vertex AI SDK
#         vertexai.init(project=project_id, location=location)
#         print(f"Core: Vertex AI SDK initialized for project '{project_id}' and location '{location}'.")

#         # Load the specific Gemini model
#         gemini_model = GenerativeModel("gemini-1.0-pro-001") # Using a standard, effective model
#         print("Core: Gemini Pro model loaded successfully.")

#     except Exception as e:
#         print(f"Core_Error: Failed to initialize one or more GCP services.")
#         print(f"Core_Error: Details - {e}")
#         print("Core_Hint: Ensure your GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly.")
#         print("Core_Hint: Make sure you have enabled 'Cloud Speech-to-Text API', 'Cloud Text-to-Speech API', and 'Vertex AI API' in your GCP project.")
#         # Ensure clients are reset to None if initialization fails
#         speech_client = tts_client = gemini_model = None
#         # Optionally, re-raise the exception to be handled by the calling application
#         # raise e

# # --- File I/O for Chat History ---

# def load_chat_history() -> List[Dict[str, Any]]:
#     """
#     Loads the chat history from the JSON file.

#     Returns:
#         List[Dict[str, Any]]: A list of chat messages, or an empty list if the file doesn't exist or is empty.
#     """
#     try:
#         if os.path.exists(CHAT_HISTORY_FILE):
#             with open(CHAT_HISTORY_FILE, 'r') as f:
#                 data = json.load(f)
#                 return data.get("items", []) # Return the list of items, or empty list if key is missing
#         return []
#     except (json.JSONDecodeError, FileNotFoundError) as e:
#         print(f"Core_Warning: Could not load chat history from {CHAT_HISTORY_FILE}. Starting fresh. Error: {e}")
#         return []

# def save_chat_history_to_file(history: List[Dict[str, Any]]):
#     """
#     Saves the updated chat history back to the JSON file.

#     Args:
#         history (List[Dict[str, Any]]): The complete list of chat messages to save.
#     """
#     try:
#         # Ensure the directory exists before trying to write the file
#         os.makedirs(os.path.dirname(CHAT_HISTORY_FILE) or '.', exist_ok=True)
#         with open(CHAT_HISTORY_FILE, 'w') as f:
#             # Store the history within a dictionary for better structure
#             json.dump({"items": history}, f, indent=4)
#     except Exception as e:
#         print(f"Core_Error: Failed to save chat history to {CHAT_HISTORY_FILE}. Error: {e}")

# # --- Core AI Pipeline Functions ---

# def transcribe_audio_gcp(audio_bytes: bytes) -> Optional[str]:
#     """
#     Transcribes the given audio bytes to text using Google Cloud Speech-to-Text.

#     Args:
#         audio_bytes (bytes): The raw audio data.

#     Returns:
#         Optional[str]: The transcribed text, or None if transcription fails.
#     """
#     if not speech_client:
#         print("Core_Error: Speech-to-Text client is not initialized.")
#         return None

#     try:
#         # Prepare the audio and configuration for the Speech-to-Text API
#         audio = speech.RecognitionAudio(content=audio_bytes)
#         # Configuration is crucial. We assume the incoming audio is WAV format (LINEAR16).
#         # Sample rate of 16000 is common for voice applications.
#         # This must match the actual recording format from the watch.
#         config = speech.RecognitionConfig(
#             encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#             sample_rate_hertz=16000,
#             language_code="en-US",  # BCP-47 language code (e.g., "en-US")
#             model="latest_long", # Use a model suited for conversation
#         )

#         print("Core: Sending audio to GCP for transcription...")
#         response = speech_client.recognize(config=config, audio=audio)
#         print("Core: Received transcription response from GCP.")

#         if response.results and response.results[0].alternatives:
#             transcript = response.results[0].alternatives[0].transcript
#             print(f"Core: Transcription successful. Text: '{transcript}'")
#             return transcript
#         else:
#             print("Core_Warning: Transcription returned no results. The audio might be silent or unclear.")
#             return None # Return None if no transcription was possible

#     except Exception as e:
#         print(f"Core_Error: An error occurred during GCP transcription: {e}")
#         traceback.print_exc()
#         return None


# def get_gemini_response(user_input: str, project_id: str, location: str) -> Optional[str]:
#     """
#     Generates a text response from Gemini based on the user's input and chat history.

#     Args:
#         user_input (str): The transcribed text from the user.
#         project_id (str): GCP project ID.
#         location (str): GCP location for Vertex AI.

#     Returns:
#         Optional[str]: The generated text response from Gemini, or None if it fails.
#     """
#     if not gemini_model:
#         # Attempt to re-initialize if the model is not ready.
#         print("Core_Warning: Gemini model not initialized. Attempting re-initialization...")
#         initialize_gcp_clients(project_id, location)
#         if not gemini_model:
#             print("Core_Error: Gemini model could not be initialized. Cannot generate response.")
#             return None

#     try:
#         # Load the existing chat history to provide context to Gemini
#         chat_history = load_chat_history()

#         # The Gemini API expects a list of alternating user and model "Parts".
#         # We need to convert our stored history to this format.
#         gemini_history = []
#         for message in chat_history:
#             role = "model" if message["role"] == "assistant" else "user"
#             gemini_history.append(Part.from_text(message["content"], role=role))
        
#         # Start a new chat session with the loaded history
#         chat = gemini_model.start_chat(history=gemini_history)

#         # The persona prompt guides Gemini on how to behave.
#         # This is critical for creating the desired user experience.
#         persona_prompt = (
#             "You are Curio, a friendly, curious, and very patient AI companion for a young child. "
#             "Your purpose is to answer questions in a simple, encouraging, and easy-to-understand way. "
#             "Always be positive and supportive. Keep your answers concise, ideally 1-3 sentences. "
#             "If you don't know an answer, say so in a simple way, like 'That's a great question! I'm not sure about that one, but we can learn together.' "
#             "Never use complex words or jargon. Your goal is to spark curiosity, not to be a walking encyclopedia."
#         )

#         # Combine the persona with the user's latest input
#         full_prompt = f"{persona_prompt}\n\nChild asks: \"{user_input}\""
        
#         print("Core: Sending prompt to Gemini...")
#         response = chat.send_message(full_prompt)
#         print("Core: Received response from Gemini.")

#         # Extract the text from the response
#         ai_response_text = response.text
#         print(f"Core: Gemini response text: '{ai_response_text}'")

#         # --- Update and Save Chat History ---
#         # Add the new user message and AI response to our history list
#         timestamp = datetime.now().isoformat()
#         chat_history.append({"role": "user", "content": user_input, "timestamp": timestamp})
#         chat_history.append({"role": "assistant", "content": ai_response_text, "timestamp": timestamp})

#         # Save the updated history back to the file
#         save_chat_history_to_file(chat_history)
#         print("Core: Chat history has been updated and saved.")
        
#         return ai_response_text

#     except Exception as e:
#         print(f"Core_Error: An error occurred while interacting with Gemini: {e}")
#         traceback.print_exc()
#         return None


# def synthesize_speech_gcp(text_input: str) -> Optional[bytes]:
#     """
#     Converts text into speech audio bytes using Google Cloud Text-to-Speech.

#     Args:
#         text_input (str): The text to be synthesized into speech.

#     Returns:
#         Optional[bytes]: The raw audio bytes of the generated speech, or None if it fails.
#     """
#     if not tts_client:
#         print("Core_Error: Text-to-Speech client is not initialized.")
#         return None

#     try:
#         # Prepare the synthesis input
#         synthesis_input = texttospeech.SynthesisInput(text=text_input)

#         # Configure the voice. These parameters can be customized for different characters.
#         # A friendly, child-like voice is chosen here.
#         # You can find more voices here: https://cloud.google.com/text-to-speech/docs/voices
#         voice = texttospeech.VoiceSelectionParams(
#             language_code="en-US",
#             name="en-US-Wavenet-G", # A good, friendly female voice
#             ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
#         )

#         # Configure the audio output format. LINEAR16 is uncompressed WAV.
#         audio_config = texttospeech.AudioConfig(
#             audio_encoding=texttospeech.AudioEncoding.LINEAR16,
#             speaking_rate=0.95, # Slightly slower for clarity for a child
#             pitch=-2.0 # Slightly higher pitch
#         )

#         print("Core: Sending text to GCP for speech synthesis...")
#         response = tts_client.synthesize_speech(
#             input=synthesis_input, voice=voice, audio_config=audio_config
#         )
#         print("Core: Received synthesized audio from GCP.")
        
#         # The response.audio_content contains the raw audio bytes
#         return response.audio_content

#     except Exception as e:
#         print(f"Core_Error: An error occurred during GCP speech synthesis: {e}")
#         traceback.print_exc()
#         return None


# # --- Main Pipeline Orchestrator ---

# def process_incoming_audio_pipeline(
#     audio_bytes: bytes,
#     project_id: str,
#     location: str
# ) -> Tuple[Optional[str], Optional[str], Optional[bytes]]:
#     """
#     The main pipeline that orchestrates the entire process:
#     1. Transcribes audio to text.
#     2. Gets an AI response based on the text.
#     3. Synthesizes the AI response back to audio.

#     Args:
#         audio_bytes (bytes): The raw audio from the watch.
#         project_id (str): Your GCP project ID.
#         location (str): The GCP location for Vertex AI.

#     Returns:
#         A tuple containing:
#         - str: The user's transcribed text.
#         - str: The AI's text response.
#         - bytes: The AI's synthesized audio response.
#         Returns (None, None, None) if any critical step fails.
#     """
#     print("\n--- Core: Starting KidWatch Audio Processing Pipeline ---")
    
#     # Step 1: Transcribe Audio to Text
#     user_transcription = transcribe_audio_gcp(audio_bytes)
#     if not user_transcription:
#         print("Core_Pipeline_Error: Transcription failed. Aborting pipeline.")
#         return None, None, None
    
#     # Step 2: Get AI Text Response from Gemini
#     ai_text_response = get_gemini_response(user_transcription, project_id, location)
#     if not ai_text_response:
#         print("Core_Pipeline_Error: Failed to get response from Gemini. Aborting pipeline.")
#         return user_transcription, None, None
        
#     # Step 3: Synthesize AI Text Response to Speech
#     ai_speech_audio_bytes = synthesize_speech_gcp(ai_text_response)
#     if not ai_speech_audio_bytes:
#         print("Core_Pipeline_Error: Speech synthesis failed. Aborting pipeline.")
#         return user_transcription, ai_text_response, None
    
#     print("--- Core: KidWatch Audio Processing Pipeline Completed Successfully ---\n")
#     return user_transcription, ai_text_response, ai_speech_audio_bytes

# # --- Standalone Test Execution ---
# # This block allows you to test the core logic directly from the command line
# # without needing to run the full FastAPI server.
# if __name__ == '__main__':
#     print("--- Running Core Voice Assistant in Standalone Test Mode ---")
    
#     # IMPORTANT: Set these environment variables before running for local tests
#     TEST_GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
#     TEST_GCP_LOCATION = os.getenv("GCP_LOCATION")
#     TEST_AUDIO_FILE_PATH = os.getenv("TEST_AUDIO_FILE_PATH") # Path to a sample .wav file

#     if not all([TEST_GCP_PROJECT_ID, TEST_GCP_LOCATION, TEST_AUDIO_FILE_PATH]):
#         print("\nError: Missing required environment variables for testing.")
#         print("Please set: GCP_PROJECT_ID, GCP_LOCATION, and TEST_AUDIO_FILE_PATH")
#         print("Example:")
#         print("export GCP_PROJECT_ID='your-project-id'")
#         print("export GCP_LOCATION='us-central1'")
#         print("export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'")
#         print("export TEST_AUDIO_FILE_PATH='./path/to/your/test_audio.wav'")
        
#     else:
#         # Initialize clients
#         initialize_gcp_clients(project_id=TEST_GCP_PROJECT_ID, location=TEST_GCP_LOCATION)
        
#         # Check if initialization was successful before proceeding
#         if speech_client and tts_client and gemini_model:
#             try:
#                 # Read the sample audio file
#                 with open(TEST_AUDIO_FILE_PATH, 'rb') as f:
#                     test_audio_bytes = f.read()
#                 print(f"\nLoaded test audio file: {TEST_AUDIO_FILE_PATH}")

#                 # Run the full pipeline
#                 transcription, ai_text, ai_audio = process_incoming_audio_pipeline(
#                     test_audio_bytes,
#                     project_id=TEST_GCP_PROJECT_ID,
#                     location=TEST_GCP_LOCATION
#                 )

#                 if transcription and ai_text and ai_audio:
#                     print("\n--- Test Results ---")
#                     print(f"User Transcription: {transcription}")
#                     print(f"AI Text Response: {ai_text}")
#                     print(f"AI Audio Generated: {len(ai_audio)} bytes")

#                     # Save the output audio to a file for verification
#                     output_audio_path = "test_output.wav"
#                     with open(output_audio_path, 'wb') as f:
#                         f.write(ai_audio)
#                     print(f"AI audio response saved to: {output_audio_path}")
#                     print("--------------------\n")
#                 else:
#                     print("\n--- Test Failed: The pipeline did not complete successfully. Check logs above. ---\n")

#             except FileNotFoundError:
#                 print(f"\nError: Test audio file not found at path: {TEST_AUDIO_FILE_PATH}")
#             except Exception as e:
#                 print(f"\nAn unexpected error occurred during the standalone test: {e}")

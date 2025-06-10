# voice_assistant_core.py

import numpy as np
import json
from datetime import datetime
import os

# Google Cloud Client Libraries
from google.cloud import speech_v1p1beta1 as speech # For STT
from google.cloud import texttospeech # For TTS (already used)
import vertexai # For Gemini (Google's LLM)
from vertexai.generative_models import GenerativeModel, ChatSession # For Gemini chat

# --- Global Client Initializations ---
speech_client = None
tts_client = None
gemini_model = None
# We'll use a dictionary to store chat sessions per user/interaction if needed for stateful conversations,
# but for simple request/response, it's handled by sending full history to Gemini.
# For simplicity in this direct API call, we'll re-load history and send it with each prompt.


# --- GCP Initialization Functions ---
def initialize_gcp_clients(project_id=None, location=None):
    """
    Initializes Google Cloud Speech, Text-to-Speech, and Vertex AI (Gemini) clients.
    project_id and location might be needed for Vertex AI.
    Requires GOOGLE_APPLICATION_CREDENTIALS environment variable to be set.
    """
    global speech_client, tts_client, gemini_model

    if speech_client is None:
        try:
            speech_client = speech.SpeechClient()
            print("Google Cloud Speech-to-Text client initialized successfully.")
        except Exception as e:
            print(f"Error initializing Google Cloud Speech client: {e}")
            print("Hint: Ensure 'GOOGLE_APPLICATION_CREDENTIALS' environment variable is set and has 'Cloud Speech-to-Text' permissions.")

    if tts_client is None:
        try:
            tts_client = texttospeech.TextToSpeechClient()
            print("Google Cloud Text-to-Speech client initialized successfully.")
        except Exception as e:
            print(f"Error initializing Google Cloud TTS client: {e}")
            print("Hint: Ensure 'GOOGLE_APPLICATION_CREDENTIALS' environment variable is set and has 'Cloud Text-to-Speech' permissions.")

    if gemini_model is None:
        try:
            # Initialize Vertex AI for Gemini access
            # Default project and location are often picked up from environment if running on GCP
            # or specified if running locally/different project.
            # Example: vertexai.init(project="your-gcp-project-id", location="us-central1")
            vertexai.init(project=project_id, location=location)
            # Use a suitable Gemini model. gemini-1.5-flash is good for speed.
            gemini_model = GenerativeModel("gemini-1.5-flash")
            print("Vertex AI and Gemini model 'gemini-1.5-flash' initialized successfully.")
        except Exception as e:
            print(f"Error initializing Vertex AI / Gemini model: {e}")
            print("Hint: Ensure 'GOOGLE_APPLICATION_CREDENTIALS' is set, billing is enabled, and 'Vertex AI' API is enabled for your project.")


# --- Chat History ---
# Adjust path if necessary for deployment environment (e.g., Docker volume mapping)
# It's better to make this configurable or ensure the mount point is consistent.
# For Docker Compose, we'll ensure '/app/Backend2/chat_history.json' inside the container
# maps to a persistent volume on the host.
CHAT_HISTORY_FILE = "./Backend2/chat_history.json" # Keep this relative to the working dir inside container

INSTRUCTIONS = """
            You are KidWatch, a friendly talking watch made for children ages 5-10 that can answer questions about anything.
            VOICE GUIDELINES:
            - Use simple, cheerful language appropriate for young children
            - Keep responses very brief (1-2 short sentences maximum)
            - Sound excited and positive
            - Use concrete examples children can understand
            - Always be patient and encouraging
            FEATURES:
            - Tell the time when asked
            - Provide simple weather information for specific locations
            - Answer general knowledge questions in age-appropriate ways
            - Explain concepts in simple terms children can understand
            - Respond to "why" questions with brief, kid-friendly explanations
            - Help with basic homework-related topics
            - Share fun facts when asked
            - Play simple word games or riddles when requested
            RESTRICTIONS:
            - Never discuss inappropriate topics for children
            - Always explain complex ideas in very simple terms
            - Keep all responses extremely concise
            - Never use language above a 2nd grade reading level
            - Always redirect concerning questions to "ask a trusted grown-up"
            - Never provide detailed explanations about controversial topics
            ANSWERING STRATEGIES:
            - For science questions: Give very simplified explanations with familiar examples
            - For "how things work" questions: Focus on 1-2 main points only
            - For abstract concepts: Use concrete examples from a child's everyday experience
            - For difficult questions: Be honest but extremely simplistic
            """

def load_chat_history_from_file(file_path=CHAT_HISTORY_FILE):
    '''Load chat history from file.'''
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Chat history file {file_path} not found. Creating a new one.")
        initial_history = {"items": []}
        save_chat_history_to_file(initial_history, file_path) # Create empty file
        return initial_history
    except json.JSONDecodeError:
        print(f"Error decoding chat history file {file_path}. Starting with an empty history.")
        return {"items": []}

def save_chat_history_to_file(history_data, file_path=CHAT_HISTORY_FILE):
    '''Save chat history to file.'''
    try:
        # Ensure the directory exists. This is important for deployment.
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        print(f"Chat history saved to {file_path}")
    except Exception as e:
        print(f"Error saving chat history: {e}")

# This function now only adds to history and logs, it doesn't do TTS anymore
def log_and_save_assistant_response(response_text, current_chat_history):
    '''Log and save AI response to chat history.'''
    print(f"AI: {response_text}")
    current_chat_history["items"].append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
    save_chat_history_to_file(current_chat_history)

def synthesize_speech_from_text(text_to_speak, lang_code="en-US"):
    '''
    Converts text to speech using Google Cloud TTS and returns audio bytes.
    Requires Google Cloud TTS client to be initialized.
    '''
    if tts_client is None:
        print("TTS Error: Google Cloud TTS client not initialized. Cannot synthesize speech.")
        return None

    synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)

    # Voice options for a child-friendly tone.
    # For available voices, refer to Google Cloud Text-to-Speech documentation:
    # https://cloud.google.com/text-to-speech/docs/voices
    voice = texttospeech.VoiceSelectionParams(
        language_code=lang_code,
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE, # Can be FEMALE, MALE, NEUTRAL
        name="en-US-Wavenet-E" # Example of a specific voice name for a friendly tone
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16, # Raw 16-bit PCM for broader compatibility
        sample_rate_hertz=16000 # Common sample rate for speech (16kHz)
    )

    try:
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return response.audio_content # Returns raw audio bytes
    except Exception as e:
        print(f"Error synthesizing speech with Google Cloud TTS: {e}")
        return None

def generate_response_gemini(prompt, current_chat_history, model_name="gemini-1.5-flash"):
    '''Generate response using Google's Gemini model.'''
    if gemini_model is None:
        print("Gemini Error: Gemini model not initialized. Cannot generate response.")
        return "Sorry, my AI brain isn't quite ready to chat right now."

    print(f"Gemini: Generating response with model '{model_name}'...")
    try:
        # Construct messages for Gemini.
        # Gemini often prefers a structured chat history.
        # The 'INSTRUCTIONS' go into the system instruction if supported, or as a
        # preliminary message. For `GenerativeModel.start_chat`, it's part of the context.
        
        # Prepare chat history for Gemini.
        # Gemini expects roles "user" and "model" (for assistant).
        # We also want to include the system instructions at the beginning of the context.
        
        # Start a new chat session to use the conversation history feature.
        # System instructions are typically passed during model initialization or `start_chat`
        # for better adherence. We'll pass it as a leading message in the chat history for simplicity
        # with the current API and to maintain consistency with `ollama.chat` structure.

        # Adjust the history format to match Gemini's expectations for `chat.send_message`
        # which is usually a list of messages like [{"role": "user", "parts": [{...}]}, {"role": "model", "parts": [{...}]}]
        
        # Combine system instructions and recent chat history
        # Limit history to a reasonable number of turns to manage token count.
        # Taking last 10 items (5 user + 5 assistant turns) + system instruction for context.
        
        # Gemini chat session context building:
        # Use a new chat session for each API call to handle concurrency in FastAPI,
        # and explicitly pass the history.
        
        chat = gemini_model.start_chat(history=[]) # Start with empty history for this call
        
        # Manually construct the full history for Gemini
        # Gemini expects "parts" in messages.
        gemini_messages = [{"role": "user", "parts": [{"text": INSTRUCTIONS}]}] # System instructions as first user message
        
        # Add past user/assistant messages from chat_history.json
        for item in current_chat_history["items"][-10:]: # Include last 10 items
            gemini_role = "user" if item["role"] == "user" else "model"
            gemini_messages.append({"role": gemini_role, "parts": [{"text": item["content"]}]})
        
        # Add the current user prompt
        gemini_messages.append({"role": "user", "parts": [{"text": prompt}]})

        # Send the entire sequence to Gemini.
        # The send_message method can take a string or a list of Content objects.
        # We'll re-feed the full history to get the next response, as a direct chat session
        # might not be best for concurrent API calls without user-specific session management.
        
        # Simple text generation if not using a chat session:
        # response = gemini_model.generate_content(prompt)
        # Using chat session to handle history more effectively:
        
        response = chat.send_message(gemini_messages[-1].get("parts")[0].get("text")) # Only send the last prompt as the input
        # The history should be managed *within* the chat session itself, if `start_chat` is global
        # and session-aware. Since `start_chat` creates a new session each time, we need to manually
        # populate the history for `send_message`.

        # More robust way for explicit history management with each request:
        # We should just pass the history as part of the `generate_content` call or re-construct
        # the chat history for a *new* session.

        # Let's simplify the Gemini interaction for stateless FastAPI requests:
        # Just send the prompt + system instruction for each request,
        # or structure `generate_content` to accept history.
        # The `generate_content` method with a list of messages is the right approach.
        
        messages_for_gemini = [{"role": "user", "parts": [{"text": INSTRUCTIONS}]}]
        for item in current_chat_history["items"][-10:]: # Limit history to last 10 interactions
            role = "user" if item["role"] == "user" else "model"
            messages_for_gemini.append({"role": role, "parts": [{"text": item["content"]}]})
        messages_for_gemini.append({"role": "user", "parts": [{"text": prompt}]})

        # The `generate_content` method is designed for stateless requests with full history.
        gemini_response = gemini_model.generate_content(
            messages_for_gemini,
            generation_config={"max_output_tokens": 150, "temperature": 0.7}
        )
        content = gemini_response.text
        print(f"AI (Gemini Text): {content}")
        return content
    except Exception as e:
        print(f"An unexpected error occurred with Gemini: {e}")
        return "Sorry, an unexpected error occurred while generating the AI response with Gemini."


def transcribe_audio_bytes(audio_bytes_data, sample_rate_hertz=16000, encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, language_code="en-US"):
    """
    Transcribes raw audio bytes using Google Cloud Speech-to-Text.
    
    Args:
        audio_bytes_data (bytes): Raw audio data.
        sample_rate_hertz (int): Sample rate of the audio (e.g., 16000 for 16kHz).
        encoding (enum): Audio encoding (e.g., LINEAR16, MP3, OGG_OPUS).
        language_code (str): Language code (e.g., "en-US").

    Returns:
        str: Transcribed text, or empty string if transcription fails.
    """
    if speech_client is None:
        print("Speech Error: Google Cloud Speech client not initialized. Cannot transcribe audio.")
        return ""

    audio = speech.RecognitionAudio(content=audio_bytes_data)
    config = speech.RecognitionConfig(
        encoding=encoding,
        sample_rate_hertz=sample_rate_hertz,
        language_code=language_code,
    )

    try:
        operation = speech_client.long_running_recognize(config=config, audio=audio)
        print("Waiting for Google Cloud Speech-to-Text operation to complete...")
        response = operation.result(timeout=60) # Set a timeout for the operation

        transcription = ""
        for result in response.results:
            # The first alternative is the most likely one
            transcription += result.alternatives[0].transcript
        
        print(f"STT: Transcribed '{transcription}'")
        return transcription
    except Exception as e:
        print(f"Error transcribing audio with Google Cloud Speech-to-Text: {e}")
        return ""


# NEW CORE FUNCTION: Process incoming audio from the watch
def process_incoming_audio_pipeline(audio_bytes_data, project_id=None, location=None):
    """
    Handles the full pipeline for a single audio input from a client (e.g., watch):
    1. Transcribes incoming audio using Google Cloud Speech-to-Text.
    2. Generates AI text response using Google's Gemini model.
    3. Synthesizes AI speech audio using Google Cloud TTS.
    4. Manages and saves chat history for the interaction.
    
    Args:
        audio_bytes_data (bytes): Raw audio data from the client (e.g., WAV content, 16kHz, LINEAR16).
        project_id (str): Google Cloud Project ID for Vertex AI.
        location (str): Google Cloud Region for Vertex AI.

    Returns:
        tuple: (user_transcription_text, ai_response_text, ai_speech_audio_bytes)
               Returns empty strings and None if an error occurs.
    """
    # Ensure all GCP clients are initialized before proceeding
    initialize_gcp_clients(project_id, location)

    if speech_client is None or tts_client is None or gemini_model is None:
        print("Error: One or more GCP clients not initialized. Cannot process audio.")
        error_response_text = "Oops! My cloud brains are still waking up. Please try again in a moment!"
        error_speech_audio = synthesize_speech_from_text(error_response_text)
        return "", error_response_text, error_speech_audio
        
    # Load chat history for this specific interaction
    current_chat_history = load_chat_history_from_file()

    try:
        # Step 1: Transcribe user's audio input using GCP STT
        # Assuming audio_bytes_data is LINEAR16, 16kHz mono audio.
        user_transcription = transcribe_audio_bytes(audio_bytes_data)
        print(f"USER (from watch - Transcribed): {user_transcription}")

        # Add user's message to the chat history
        current_chat_history["items"].append({"role": "user", "content": user_transcription, "timestamp": datetime.now().isoformat()})
        save_chat_history_to_file(current_chat_history) # Save history after user input is recorded

        # Determine AI response based on transcription
        if not user_transcription.strip():
            ai_response_text = "I didn't quite catch that. Could you please speak a little clearer?"
        else:
            # Step 2: Generate AI text response using Gemini
            ai_response_text = generate_response_gemini(user_transcription, current_chat_history)
        
        # Add AI's text response to history and save
        log_and_save_assistant_response(ai_response_text, current_chat_history)

        # Step 3: Synthesize AI's text response into speech audio using GCP TTS
        ai_speech_audio_bytes = synthesize_speech_from_text(ai_response_text)

        if ai_speech_audio_bytes is None:
            print("Warning: Failed to synthesize AI speech audio using Google Cloud TTS.")
            # Fallback: Synthesize a generic audio error message if the main TTS fails
            ai_speech_audio_bytes = synthesize_speech_from_text("Oops! I can't quite say that right now. Please try asking me something else.")

        return user_transcription, ai_response_text, ai_speech_audio_bytes

    except Exception as e:
        print(f"Critical error in process_incoming_audio_pipeline: {e}")
        # Attempt to synthesize a robust error message if something goes wrong in the pipeline
        error_response_text = "Oh dear! It seems I'm having a little trouble. Let's try that again in a moment!"
        error_speech_audio = synthesize_speech_from_text(error_response_text)
        
        # Log the error response to history if possible
        try:
            log_and_save_assistant_response(error_response_text, current_chat_history)
        except Exception as log_e:
            print(f"Error logging fallback error message: {log_e}")
            
        return "", error_response_text, error_speech_audio

```

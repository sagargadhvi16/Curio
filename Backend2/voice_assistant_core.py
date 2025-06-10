# voice_assistant_core.py

import webrtcvad # Kept for potential future server-side VAD, but not used in the new main pipeline
import pyaudio # Kept for potential future server-side VAD, but not used in the new main pipeline
import whisper
import numpy as np
from ollama import Options
import ollama
import pyttsx3 # This will be replaced for cloud TTS for API responses, but might be kept for local testing
import torch
import json
from datetime import datetime
import time
from google.cloud import texttospeech # New import for Google Cloud TTS
import os # For accessing environment variables like GOOGLE_APPLICATION_CREDENTIALS

# --- Whisper Model Initialization ---
WHISPER_MODEL_NAME = "base.en" # Using .en for english-only can be faster/more accurate
WHISPER_DEVICE = None

if torch.cuda.is_available():
    WHISPER_DEVICE = "cuda"
    print(f"Whisper: Using CUDA (NVIDIA GPU: {torch.cuda.get_device_name(0)})")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    WHISPER_DEVICE = "mps"
    print("Whisper: Using MPS (Apple Silicon GPU)")
else:
    WHISPER_DEVICE = "cpu"
    print("Whisper: CUDA and MPS not available, using CPU for Whisper.")

# Global stt_model, to be loaded by the API endpoint or application startup
stt_model = None

# --- Google Cloud TTS Initialization ---
tts_client = None

def initialize_tts_client():
    """
    Initializes the Google Cloud Text-to-Speech client.
    Requires GOOGLE_APPLICATION_CREDENTIALS environment variable to be set.
    """
    global tts_client
    if tts_client is None:
        try:
            # Ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set
            # to the path of your service account key JSON file.
            tts_client = texttospeech.TextToSpeechClient()
            print("Google Cloud Text-to-Speech client initialized successfully.")
        except Exception as e:
            print(f"Error initializing Google Cloud TTS client: {e}")
            print("Hint: Ensure 'GOOGLE_APPLICATION_CREDENTIALS' environment variable is set to your service account key file path.")

# --- Chat History ---
# Adjust path if necessary for deployment environment (e.g., Docker volume mapping)
CHAT_HISTORY_FILE = "./Backend2/chat_history.json"

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
            TOOL USAGE: (Note: Tools are not implemented in this version but kept for LLM context)
            - Use the generic_lookup tool with appropriate query_type parameter:
              - "time" for telling the time (no query parameter needed)
              - "weather" for weather info (use location as query)
              - "fact" for interesting facts (use topic as query)
              - "joke" for kid-friendly jokes (no query parameter needed)
              - "riddle" for simple riddles (no query parameter needed)
              - For other types of information, use an appropriate descriptive query_type
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
    initialize_tts_client() # Ensure client is initialized
    if tts_client is None:
        print("TTS Error: Google Cloud TTS client not initialized. Cannot synthesize speech.")
        return None

    synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)

    # Voice options for a child-friendly tone.
    # For available voices, refer to Google Cloud Text-to-Speech documentation:
    # https://cloud.google.com/text-to-speech/docs/voices
    # Examples:
    # 'en-US-Wavenet-E' (female, natural)
    # 'en-US-Wavenet-D' (male, natural)
    # 'en-US-Standard-E' (female, standard)
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


def generate_response_ollama(prompt, current_chat_history, model_name='llama3.2', max_response_tokens=150, temperature=0.7):
    '''Generate response using OLLAMA.'''
    print(f"Ollama: Generating response with model '{model_name}'...")
    try:
        # Construct messages for Ollama, including system instructions and recent chat history
        # Limiting history to the last few exchanges to manage token count and focus context.
        # Here, taking the last 10 messages (5 user + 5 assistant turns).
        recent_history_for_ollama = current_chat_history["items"][-10:]
        
        messages = [{"role": "system", "content": INSTRUCTIONS}] + recent_history_for_ollama + [{"role": "user", "content": prompt}]

        response = ollama.chat(
            model=model_name,
            messages=messages,
            options=Options(
                num_predict=max_response_tokens,
                temperature=temperature,
            )
        )
        content = response['message']['content']
        print(f"AI (Ollama Text): {content}")
        return content
    except ollama.ResponseError as e:
        print(f"Ollama API Error: {e.error}")
        if "model not found" in e.error.lower():
            print(f"Hint: Model '{model_name}' not found. Pull it with 'ollama pull {model_name}' or check available models with 'ollama list'.")
        return f"Sorry, there was an API error with the AI model: {e.error}"
    except ollama.RequestError as e:
        print(f"Ollama Connection Error: {e}")
        print("Hint: Ensure the Ollama server is running and accessible. Download from https://ollama.com/download")
        return "Sorry, I couldn't connect to the Ollama AI service. Please check if it's running."
    except Exception as e:
        print(f"An unexpected error occurred with Ollama: {e}")
        return "Sorry, an unexpected error occurred while generating the AI response."

# VAD and PyAudio related global variables are kept but their usage in the
# `record_and_transcribe_loop` is removed. They are not directly used in the new
# `process_incoming_audio_pipeline` but might be relevant if you later implement
# server-side VAD for continuous streams (which is more advanced).
# vad = webrtcvad.Vad()
# vad.set_mode(3)
# RATE = 16000
# FRAME_DURATION_MS = 30
# CHUNK = int(RATE * FRAME_DURATION_MS / 1000)


def transcribe_audio_bytes(audio_bytes_data, whisper_model_instance, current_whisper_device):
    '''
    Transcribes raw audio bytes using the loaded Whisper model.
    Assumes audio_bytes_data is 16-bit PCM, 16kHz mono audio.
    '''
    # Convert raw bytes (assuming 16-bit PCM) to float32 numpy array for Whisper
    # 32768.0 is 2^15, for normalizing 16-bit signed PCM to -1.0 to 1.0 range
    audio_array = np.frombuffer(audio_bytes_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    use_fp16_whisper = (current_whisper_device == "cuda")
    print(f"Whisper: Transcribing audio bytes (fp16: {use_fp16_whisper})...")
    # Specify language="en" for faster and more accurate English-only transcription
    result = whisper_model_instance.transcribe(audio_array, fp16=use_fp16_whisper, language="en")
    return result["text"]

# NEW CORE FUNCTION: Process incoming audio from the watch
def process_incoming_audio_pipeline(audio_bytes_data, stt_model_instance, current_whisper_device):
    """
    Handles the full pipeline for a single audio input from a client (e.g., watch):
    1. Transcribes incoming audio using Whisper.
    2. Generates AI text response using Ollama.
    3. Synthesizes AI speech audio using Google Cloud TTS.
    4. Manages and saves chat history for the interaction.
    
    Args:
        audio_bytes_data (bytes): Raw audio data from the client (e.g., WAV content).
        stt_model_instance: The loaded Whisper STT model instance.
        current_whisper_device (str): The device Whisper is running on ('cpu', 'cuda', 'mps').

    Returns:
        tuple: (user_transcription_text, ai_response_text, ai_speech_audio_bytes)
               Returns empty strings and None if an error occurs.
    """
    if stt_model_instance is None:
        print("Error: Whisper STT model not loaded. Cannot process audio.")
        # Attempt to provide an audio error response
        error_response_text = "Sorry, my brain isn't quite ready yet. Please try again in a moment."
        error_speech_audio = synthesize_speech_from_text(error_response_text)
        return "", error_response_text, error_speech_audio
        
    # Load chat history for this specific interaction
    current_chat_history = load_chat_history_from_file()

    try:
        # Step 1: Transcribe user's audio input
        user_transcription = transcribe_audio_bytes(audio_bytes_data, stt_model_instance, current_whisper_device)
        print(f"USER (from watch - Transcribed): {user_transcription}")

        # Add user's message to the chat history
        current_chat_history["items"].append({"role": "user", "content": user_transcription, "timestamp": datetime.now().isoformat()})
        save_chat_history_to_file(current_chat_history) # Save history after user input is recorded

        # Determine AI response based on transcription
        if not user_transcription.strip():
            ai_response_text = "I didn't quite catch that. Could you please speak a little clearer?"
        else:
            # Step 2: Generate AI text response using Ollama
            ai_response_text = generate_response_ollama(user_transcription, current_chat_history)
        
        # Add AI's text response to history and save
        log_and_save_assistant_response(ai_response_text, current_chat_history)

        # Step 3: Synthesize AI's text response into speech audio
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

# The original `run_assistant_process` (which contained the PyAudio listening loop)
# is now conceptually removed as the backend transitions to an API-driven audio input model.
# The loading of `stt_model` will need to be handled during the FastAPI application's startup.
```





# voice_assistant_core.py

# import webrtcvad
# import pyaudio
# import whisper
# import numpy as np
# from ollama import Options
# import ollama
# import pyttsx3
# # import simpleaudio as sa # Not used
# import torch
# import json
# from datetime import datetime
# import time # For checking stop event periodically

# # --- Whisper Model Initialization (Keep as is) ---
# WHISPER_MODEL_NAME = "base.en" # Using .en for english-only can be faster/more accurate
# WHISPER_DEVICE = None

# if torch.cuda.is_available():
#     WHISPER_DEVICE = "cuda"
#     print(f"Whisper: Using CUDA (NVIDIA GPU: {torch.cuda.get_device_name(0)})")
# elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#     WHISPER_DEVICE = "mps"
#     print("Whisper: Using MPS (Apple Silicon GPU)")
# else:
#     WHISPER_DEVICE = "cpu"
#     print("Whisper: CUDA and MPS not available, using CPU for Whisper.")

# # Global stt_model, to be loaded by the process that runs the assistant
# stt_model = None

# # --- Chat History (Keep as is, but ensure functions are easily callable) ---
# CHAT_HISTORY_FILE = "./Backend2/chat_history.json"
# # chat_history will be loaded and managed within the running assistant instance

# INSTRUCTIONS = """
#             You are KidWatch, a friendly talking watch made for children ages 5-10 that can answer questions about anything.
#             VOICE GUIDELINES:
#             - Use simple, cheerful language appropriate for young children
#             - Keep responses very brief (1-2 short sentences maximum)
#             - Sound excited and positive
#             - Use concrete examples children can understand
#             - Always be patient and encouraging
#             FEATURES:
#             - Tell the time when asked
#             - Provide simple weather information for specific locations
#             - Answer general knowledge questions in age-appropriate ways
#             - Explain concepts in simple terms children can understand
#             - Respond to "why" questions with brief, kid-friendly explanations
#             - Help with basic homework-related topics
#             - Share fun facts when asked
#             - Play simple word games or riddles when requested
#             TOOL USAGE:
#             - Use the generic_lookup tool with appropriate query_type parameter:
#               - "time" for telling the time (no query parameter needed)
#               - "weather" for weather info (use location as query)
#               - "fact" for interesting facts (use topic as query)
#               - "joke" for kid-friendly jokes (no query parameter needed)
#               - "riddle" for simple riddles (no query parameter needed)
#               - For other types of information, use an appropriate descriptive query_type
#             RESTRICTIONS:
#             - Never discuss inappropriate topics for children
#             - Always explain complex ideas in very simple terms
#             - Keep all responses extremely concise
#             - Never use language above a 2nd grade reading level
#             - Always redirect concerning questions to "ask a trusted grown-up"
#             - Never provide detailed explanations about controversial topics
#             ANSWERING STRATEGIES:
#             - For science questions: Give very simplified explanations with familiar examples
#             - For "how things work" questions: Focus on 1-2 main points only
#             - For abstract concepts: Use concrete examples from a child's everyday experience
#             - For difficult questions: Be honest but extremely simplistic
#             """

# def load_chat_history_from_file(file_path=CHAT_HISTORY_FILE):
#     '''Load chat history from file.'''
#     try:
#         with open(file_path, 'r') as f:
#             return json.load(f)
#     except FileNotFoundError:
#         return {"items": []}
#     except json.JSONDecodeError:
#         print(f"Error decoding chat history file {file_path}. Starting with an empty history.")
#         return {"items": []}

# def save_chat_history_to_file(history_data, file_path=CHAT_HISTORY_FILE):
#     '''Save chat history to file.'''
#     try:
#         with open(file_path, 'w') as f:
#             json.dump(history_data, f, indent=2)
#         print(f"Chat history saved to {file_path}")
#     except Exception as e:
#         print(f"Error saving chat history: {e}")


# # Make chat_history a parameter for functions that modify it
# def speak_response(response_text, current_chat_history, lang="en"):
#     '''Convert AI response to speech using pyttsx3.'''
#     try:
#         engine = pyttsx3.init()
#         engine.say(response_text)
#         engine.runAndWait()
#         current_chat_history["items"].append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
#     except Exception as e:
#         print(f"Error in TTS: {e}")

# def generate_response_ollama(prompt, current_chat_history, model_name='llama3.2', max_response_tokens=150, temperature=0.7):
#     '''Generate response using OLLAMA.'''
#     print(f"Ollama: Generating response with model '{model_name}'...")
#     try:
#         # messages = [{"role": "system", "content": INSTRUCTIONS}] + current_chat_history["items"] + [{"role": "user", "content": prompt}]
#         messages = [{"role": "system", "content": INSTRUCTIONS}] + [{"role": "user", "content": prompt}]
#         response = ollama.chat(
#             model=model_name,
#             messages=messages,
#             options=Options(
#                 num_predict=max_response_tokens,
#                 temperature=temperature,
#             )
#         )
#         content = response['message']['content']
#         print(f"AI: {content}")
#         return content
#     except ollama.ResponseError as e:
#         print(f"Ollama API Error: {e.error}")
#         if "model not found" in e.error.lower():
#             print(f"Hint: Model '{model_name}' not found. Pull it with 'ollama pull {model_name}' or check available models with 'ollama list'.")
#         return f"Sorry, there was an API error with the AI model: {e.error}"
#     except ollama.RequestError as e:
#         print(f"Ollama Connection Error: {e}")
#         print("Hint: Ensure the Ollama server is running and accessible. Download from https://ollama.com/download")
#         return "Sorry, I couldn't connect to the Ollama AI service. Please check if it's running."
#     except Exception as e:
#         print(f"An unexpected error occurred with Ollama: {e}")
#         return "Sorry, an unexpected error occurred while generating the AI response."

# # VAD and Audio Params (Keep as is)
# vad = webrtcvad.Vad()
# vad.set_mode(3)
# RATE = 16000
# FRAME_DURATION_MS = 30
# CHUNK = int(RATE * FRAME_DURATION_MS / 1000)

# def transcribe_audio_chunk(frames, whisper_model_instance, current_whisper_device):
#     '''Transcribe audio frames using the loaded Whisper model.'''
#     audio_data_bytes = b"".join(frames)
#     audio_array = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
#     use_fp16_whisper = (current_whisper_device == "cuda")
#     # print(f"Whisper: Transcribing audio chunk (fp16: {use_fp16_whisper})...") # reduce verbosity
#     result = whisper_model_instance.transcribe(audio_array, fp16=use_fp16_whisper, language="en") # Specify language
#     return result["text"]

# def is_speech(frame_data, sample_rate):
#     '''Check if a frame contains speech using VAD.'''
#     try:
#         return vad.is_speech(frame_data, sample_rate)
#     except Exception:
#         return False

# # Modified record_and_transcribe to accept a stop_event and manage its own chat_history
# def record_and_transcribe_loop(stop_event, whisper_model_instance, current_whisper_device):
#     '''
#     Records audio, detects speech, transcribes, and generates responses.
#     Accepts a stop_event to allow graceful termination.
#     Manages its own chat_history instance.
#     '''
#     # Load chat history for this session
#     current_chat_history = load_chat_history_from_file()
#     if not current_chat_history["items"]:
#         initial_message = "Hi there! I'm KidWatch, your super cool talking watch! What fun thing shall we talk about today?"
#         print("KidWatch (Initial):", initial_message)
#         # Speak response will also add to current_chat_history
#         speak_response(initial_message, current_chat_history)
#         # Save immediately after initial greeting so it's not repeated if restarted quickly
#         save_chat_history_to_file(current_chat_history)


#     audio = pyaudio.PyAudio()
#     stream = None
#     try:
#         stream = audio.open(format=pyaudio.paInt16,
#                             channels=1,
#                             rate=RATE,
#                             input=True,
#                             frames_per_buffer=CHUNK)
#     except Exception as e:
#         print(f"Error opening audio stream: {e}")
#         # Potentially log this to a file or send status back if more advanced IPC is used
#         if "PortAudio" in str(e) or "No Default Input Device Available" in str(e):
#              print("Hint: This often means no microphone is detected or PyAudio is not set up correctly.")
#         if audio: audio.terminate()
#         return # Cannot proceed without audio

#     print("KidWatch: Listening... (Speak to activate)")

#     frames_for_transcription = []
#     silence_counter_ms = 0
#     MIN_SILENCE_MS_TO_PROCESS = 1500
#     is_currently_speaking = False
#     speech_started = False

#     try:
#         while not stop_event.is_set(): # Check the stop event in the loop
#             try:
#                 frame = stream.read(CHUNK, exception_on_overflow=False)
#             except IOError as e:
#                 if e.errno == pyaudio.paInputOverflowed:
#                     print("Warning: Input overflowed. Some audio data may have been lost.")
#                     continue
#                 else:
#                     print(f"Stream read IO Error: {e}")
#                     stop_event.set() # Signal stop on critical error
#                     break
#             except Exception as e: # Catch other potential stream errors
#                 print(f"Unexpected stream error: {e}")
#                 stop_event.set()
#                 break


#             if is_speech(frame, RATE):
#                 if not speech_started:
#                     print("KidWatch: Speech detected, recording...")
#                     speech_started = True
#                 is_currently_speaking = True
#                 silence_counter_ms = 0
#                 frames_for_transcription.append(frame)
#             else:
#                 if is_currently_speaking:
#                     silence_counter_ms += FRAME_DURATION_MS
#                     frames_for_transcription.append(frame)

#                     if silence_counter_ms >= MIN_SILENCE_MS_TO_PROCESS:
#                         print("KidWatch: Silence threshold reached, processing audio...")

#                         if not frames_for_transcription:
#                             print("KidWatch: No frames to transcribe despite reaching silence threshold.")
#                             # Reset state and continue
#                             frames_for_transcription = []
#                             silence_counter_ms = 0
#                             is_currently_speaking = False
#                             speech_started = False
#                             print("KidWatch: Listening...")
#                             continue

#                         transcription = transcribe_audio_chunk(frames_for_transcription, whisper_model_instance, current_whisper_device)
#                         print(f"USER: {transcription}")

#                         current_chat_history["items"].append({"role": "user", "content": transcription, "timestamp": datetime.now().isoformat()})

#                         if transcription.strip():
#                             ai_response = generate_response_ollama(transcription, current_chat_history)
#                             if ai_response:
#                                 speak_response(ai_response, current_chat_history)
#                         else:
#                             print("KidWatch: Transcription was empty, skipping AI response.")

#                         frames_for_transcription = []
#                         silence_counter_ms = 0
#                         is_currently_speaking = False
#                         speech_started = False
#                         print("KidWatch: Listening...")
            
#             # Add a small sleep to prevent busy-waiting and allow the stop_event to be checked more readily
#             # This also makes it easier for the GIL to be released if this were a thread (though it's a process)
#             time.sleep(0.01)


#     except Exception as e: # Catch any unexpected errors in the main loop
#         print(f"An error occurred in the main assistant loop: {e}")
#     finally:
#         print("KidWatch: Stopping...")
#         if stream and stream.is_active():
#             stream.stop_stream()
#             stream.close()
#         if audio:
#             audio.terminate()
#         # Save history when the loop is exited (either by stop_event or error)
#         save_chat_history_to_file(current_chat_history)
#         print("KidWatch: Cleanup complete. Process should now exit.")

# # This function will be the target for the multiprocessing.Process
# def run_assistant_process(stop_event):
#     global stt_model # Allow modification of the global stt_model for this process
#     print("KidWatch Process: Initializing...")
#     print(f"KidWatch Process: Whisper STT will run on: {WHISPER_DEVICE.upper()}")
#     print(f"KidWatch Process: Ollama server should be running for AI responses.")

#     try:
#         print(f"KidWatch Process: Whisper loading model '{WHISPER_MODEL_NAME}' on device '{WHISPER_DEVICE}'...")
#         stt_model = whisper.load_model(WHISPER_MODEL_NAME, device=WHISPER_DEVICE)
#         print(f"KidWatch Process: Whisper model '{WHISPER_MODEL_NAME}' loaded successfully on '{WHISPER_DEVICE}'.")
#         if WHISPER_DEVICE == "cuda":
#             print(f"KidWatch Process: Whisper FP16 will be used for transcription on CUDA device.")
#     except Exception as e:
#         print(f"KidWatch Process: Error loading Whisper model on '{WHISPER_DEVICE}': {e}")
#         print("KidWatch Process: Whisper falling back to CPU.")
#         fallback_device = "cpu"
#         try:
#             stt_model = whisper.load_model(WHISPER_MODEL_NAME, device=fallback_device)
#             print(f"KidWatch Process: Whisper model '{WHISPER_MODEL_NAME}' loaded successfully on CPU (fallback).")
#             # Update WHISPER_DEVICE if fallback successful, for transcribe_audio_chunk
#             # This change is local to this process.
#             # If you need to communicate this back, you'd need a Pipe or Queue.
#             # For now, just ensure transcribe_audio_chunk uses the correct device.
#             # The global WHISPER_DEVICE in this process scope will be updated implicitly if needed.
#         except Exception as e_cpu:
#             print(f"KidWatch Process: CRITICAL - Failed to load Whisper model on CPU as fallback: {e_cpu}")
#             print("KidWatch Process: Assistant cannot start without STT model. Exiting process.")
#             return # Exit if model loading fails completely

#     # Check if stt_model was successfully loaded
#     if stt_model is None:
#         print("KidWatch Process: STT model not loaded. Assistant cannot run. Exiting process.")
#         return

#     record_and_transcribe_loop(stop_event, stt_model, WHISPER_DEVICE)
#     print("KidWatch Process: Exited record_and_transcribe_loop.")

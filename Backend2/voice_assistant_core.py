# voice_assistant_core.py

import webrtcvad
import pyaudio
import whisper
import numpy as np
from ollama import Options
import ollama
import pyttsx3
# import simpleaudio as sa # Not used
import torch
import json
from datetime import datetime
import time # For checking stop event periodically

# --- Whisper Model Initialization (Keep as is) ---
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

# Global stt_model, to be loaded by the process that runs the assistant
stt_model = None

# --- Chat History (Keep as is, but ensure functions are easily callable) ---
CHAT_HISTORY_FILE = "./Backend2/chat_history.json"
# chat_history will be loaded and managed within the running assistant instance

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
            TOOL USAGE:
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
        return {"items": []}
    except json.JSONDecodeError:
        print(f"Error decoding chat history file {file_path}. Starting with an empty history.")
        return {"items": []}

def save_chat_history_to_file(history_data, file_path=CHAT_HISTORY_FILE):
    '''Save chat history to file.'''
    try:
        with open(file_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        print(f"Chat history saved to {file_path}")
    except Exception as e:
        print(f"Error saving chat history: {e}")


# Make chat_history a parameter for functions that modify it
def speak_response(response_text, current_chat_history, lang="en"):
    '''Convert AI response to speech using pyttsx3.'''
    try:
        engine = pyttsx3.init()
        engine.say(response_text)
        engine.runAndWait()
        current_chat_history["items"].append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        print(f"Error in TTS: {e}")

def generate_response_ollama(prompt, current_chat_history, model_name='llama3.2', max_response_tokens=150, temperature=0.7):
    '''Generate response using OLLAMA.'''
    print(f"Ollama: Generating response with model '{model_name}'...")
    try:
        # messages = [{"role": "system", "content": INSTRUCTIONS}] + current_chat_history["items"] + [{"role": "user", "content": prompt}]
        messages = [{"role": "system", "content": INSTRUCTIONS}] + [{"role": "user", "content": prompt}]
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options=Options(
                num_predict=max_response_tokens,
                temperature=temperature,
            )
        )
        content = response['message']['content']
        print(f"AI: {content}")
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

# VAD and Audio Params (Keep as is)
vad = webrtcvad.Vad()
vad.set_mode(3)
RATE = 16000
FRAME_DURATION_MS = 30
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)

def transcribe_audio_chunk(frames, whisper_model_instance, current_whisper_device):
    '''Transcribe audio frames using the loaded Whisper model.'''
    audio_data_bytes = b"".join(frames)
    audio_array = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    use_fp16_whisper = (current_whisper_device == "cuda")
    # print(f"Whisper: Transcribing audio chunk (fp16: {use_fp16_whisper})...") # reduce verbosity
    result = whisper_model_instance.transcribe(audio_array, fp16=use_fp16_whisper, language="en") # Specify language
    return result["text"]

def is_speech(frame_data, sample_rate):
    '''Check if a frame contains speech using VAD.'''
    try:
        return vad.is_speech(frame_data, sample_rate)
    except Exception:
        return False

# Modified record_and_transcribe to accept a stop_event and manage its own chat_history
def record_and_transcribe_loop(stop_event, whisper_model_instance, current_whisper_device):
    '''
    Records audio, detects speech, transcribes, and generates responses.
    Accepts a stop_event to allow graceful termination.
    Manages its own chat_history instance.
    '''
    # Load chat history for this session
    current_chat_history = load_chat_history_from_file()
    if not current_chat_history["items"]:
        initial_message = "Hi there! I'm KidWatch, your super cool talking watch! What fun thing shall we talk about today?"
        print("KidWatch (Initial):", initial_message)
        # Speak response will also add to current_chat_history
        speak_response(initial_message, current_chat_history)
        # Save immediately after initial greeting so it's not repeated if restarted quickly
        save_chat_history_to_file(current_chat_history)


    audio = pyaudio.PyAudio()
    stream = None
    try:
        stream = audio.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"Error opening audio stream: {e}")
        # Potentially log this to a file or send status back if more advanced IPC is used
        if "PortAudio" in str(e) or "No Default Input Device Available" in str(e):
             print("Hint: This often means no microphone is detected or PyAudio is not set up correctly.")
        if audio: audio.terminate()
        return # Cannot proceed without audio

    print("KidWatch: Listening... (Speak to activate)")

    frames_for_transcription = []
    silence_counter_ms = 0
    MIN_SILENCE_MS_TO_PROCESS = 1500
    is_currently_speaking = False
    speech_started = False

    try:
        while not stop_event.is_set(): # Check the stop event in the loop
            try:
                frame = stream.read(CHUNK, exception_on_overflow=False)
            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed:
                    print("Warning: Input overflowed. Some audio data may have been lost.")
                    continue
                else:
                    print(f"Stream read IO Error: {e}")
                    stop_event.set() # Signal stop on critical error
                    break
            except Exception as e: # Catch other potential stream errors
                print(f"Unexpected stream error: {e}")
                stop_event.set()
                break


            if is_speech(frame, RATE):
                if not speech_started:
                    print("KidWatch: Speech detected, recording...")
                    speech_started = True
                is_currently_speaking = True
                silence_counter_ms = 0
                frames_for_transcription.append(frame)
            else:
                if is_currently_speaking:
                    silence_counter_ms += FRAME_DURATION_MS
                    frames_for_transcription.append(frame)

                    if silence_counter_ms >= MIN_SILENCE_MS_TO_PROCESS:
                        print("KidWatch: Silence threshold reached, processing audio...")

                        if not frames_for_transcription:
                            print("KidWatch: No frames to transcribe despite reaching silence threshold.")
                            # Reset state and continue
                            frames_for_transcription = []
                            silence_counter_ms = 0
                            is_currently_speaking = False
                            speech_started = False
                            print("KidWatch: Listening...")
                            continue

                        transcription = transcribe_audio_chunk(frames_for_transcription, whisper_model_instance, current_whisper_device)
                        print(f"USER: {transcription}")

                        current_chat_history["items"].append({"role": "user", "content": transcription, "timestamp": datetime.now().isoformat()})

                        if transcription.strip():
                            ai_response = generate_response_ollama(transcription, current_chat_history)
                            if ai_response:
                                speak_response(ai_response, current_chat_history)
                        else:
                            print("KidWatch: Transcription was empty, skipping AI response.")

                        frames_for_transcription = []
                        silence_counter_ms = 0
                        is_currently_speaking = False
                        speech_started = False
                        print("KidWatch: Listening...")
            
            # Add a small sleep to prevent busy-waiting and allow the stop_event to be checked more readily
            # This also makes it easier for the GIL to be released if this were a thread (though it's a process)
            time.sleep(0.01)


    except Exception as e: # Catch any unexpected errors in the main loop
        print(f"An error occurred in the main assistant loop: {e}")
    finally:
        print("KidWatch: Stopping...")
        if stream and stream.is_active():
            stream.stop_stream()
            stream.close()
        if audio:
            audio.terminate()
        # Save history when the loop is exited (either by stop_event or error)
        save_chat_history_to_file(current_chat_history)
        print("KidWatch: Cleanup complete. Process should now exit.")

# This function will be the target for the multiprocessing.Process
def run_assistant_process(stop_event):
    global stt_model # Allow modification of the global stt_model for this process
    print("KidWatch Process: Initializing...")
    print(f"KidWatch Process: Whisper STT will run on: {WHISPER_DEVICE.upper()}")
    print(f"KidWatch Process: Ollama server should be running for AI responses.")

    try:
        print(f"KidWatch Process: Whisper loading model '{WHISPER_MODEL_NAME}' on device '{WHISPER_DEVICE}'...")
        stt_model = whisper.load_model(WHISPER_MODEL_NAME, device=WHISPER_DEVICE)
        print(f"KidWatch Process: Whisper model '{WHISPER_MODEL_NAME}' loaded successfully on '{WHISPER_DEVICE}'.")
        if WHISPER_DEVICE == "cuda":
            print(f"KidWatch Process: Whisper FP16 will be used for transcription on CUDA device.")
    except Exception as e:
        print(f"KidWatch Process: Error loading Whisper model on '{WHISPER_DEVICE}': {e}")
        print("KidWatch Process: Whisper falling back to CPU.")
        fallback_device = "cpu"
        try:
            stt_model = whisper.load_model(WHISPER_MODEL_NAME, device=fallback_device)
            print(f"KidWatch Process: Whisper model '{WHISPER_MODEL_NAME}' loaded successfully on CPU (fallback).")
            # Update WHISPER_DEVICE if fallback successful, for transcribe_audio_chunk
            # This change is local to this process.
            # If you need to communicate this back, you'd need a Pipe or Queue.
            # For now, just ensure transcribe_audio_chunk uses the correct device.
            # The global WHISPER_DEVICE in this process scope will be updated implicitly if needed.
        except Exception as e_cpu:
            print(f"KidWatch Process: CRITICAL - Failed to load Whisper model on CPU as fallback: {e_cpu}")
            print("KidWatch Process: Assistant cannot start without STT model. Exiting process.")
            return # Exit if model loading fails completely

    # Check if stt_model was successfully loaded
    if stt_model is None:
        print("KidWatch Process: STT model not loaded. Assistant cannot run. Exiting process.")
        return

    record_and_transcribe_loop(stop_event, stt_model, WHISPER_DEVICE)
    print("KidWatch Process: Exited record_and_transcribe_loop.")
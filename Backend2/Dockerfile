# Use a slim Python 3.9 image as the base for a smaller footprint
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for some Python packages (like PyAudio/webrtcvad-wheels,
# even if not directly used for live mic input, their build dependencies are often needed).
# Also install 'git' if you later use git-based dependencies in requirements.txt.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    portaudio19-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the working directory first.
# This allows Docker to cache the pip install step if requirements.txt doesn't change,
# speeding up subsequent builds.
COPY Backend2/requirements.txt .

# Install Python dependencies from requirements.txt
# --no-cache-dir reduces the image size.
RUN pip install --no-cache-dir -r requirements.txt

# Create the directory for chat history.
# This ensures the directory exists before the application tries to write to it.
# The `os.makedirs` in voice_assistant_core will handle sub-directories.
RUN mkdir -p Backend2/database/chat_sessions

# Copy all your application code from the Backend2 directory into the container's /app
# This copies api.py, voice_assistant_core.py, interest_analysis.py, and chat_history.json
# Ensure this COPY command's source path correctly reflects your project structure.
COPY Backend2/ .

# Expose the port that FastAPI will run on.
# This makes the port accessible from outside the container.
EXPOSE 8000

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable.
# IMPORTANT: When you run this container, you MUST provide your Google Cloud
# service account key file and mount it into the container at this path.
# Example Docker run command argument: -v /path/to/your/local/key.json:/app/google-cloud-key.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/google-cloud-key.json

# Define the command to run your FastAPI application using Uvicorn.
# The --host 0.0.0.0 makes the server accessible from any network interface inside the container,
# which is necessary for external access through the exposed port.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# --- IMPORTANT NOTE REGARDING OLLAMA ---
# This Dockerfile is designed specifically for your FastAPI application and its dependencies.
# It does NOT include the Ollama server itself.
#
# For your FastAPI application to function correctly, the Ollama server must be:
# 1. Running as a separate process on your host VM.
#    (You would install Ollama natively on the VM and pull your 'llama3.2' model).
# OR
# 2. Running in its own separate Docker container.
#    (This is the recommended approach for production, typically managed by 'docker-compose').
#
# Your FastAPI container will need to be able to access the Ollama server.
# If Ollama is on the same VM/Docker network, it can usually be reached via 'localhost:11434'
# or by its service name if using Docker Compose.

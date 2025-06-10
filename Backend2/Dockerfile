FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Install general build essentials if any Python packages need compilation
# removed portaudio19-dev and libatlas-base-dev as they are not needed for GCP services
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the working directory first.
# This allows Docker to cache the pip install step if requirements.txt doesn't change.
COPY Backend2/requirements.txt .

# Install Python dependencies from requirements.txt
# --no-cache-dir reduces the image size.
RUN pip install --no-cache-dir -r requirements.txt

# Create the directory for chat history.
# This ensures the directory exists before the application tries to write to it.
RUN mkdir -p Backend2/database/chat_sessions

# Copy all your application code from the Backend2 directory into the container's /app
COPY Backend2/ .

# Expose the port that FastAPI will run on.
EXPOSE 8000

# Define build arguments for GCP Project ID and Location.
# These will be passed during the `docker build` command.
ARG GCP_PROJECT_ID
ARG GCP_LOCATION

# Set environment variables inside the container for FastAPI app to use.
# These will be used by vertexai.init() in voice_assistant_core.py
ENV GCP_PROJECT_ID=${GCP_PROJECT_ID}
ENV GCP_LOCATION=${GCP_LOCATION}

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable.
# IMPORTANT: When you run this container, you MUST provide your Google Cloud
# service account key file and mount it into the container at this specific path.
# Example in docker-compose.yml will handle this.
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/google-cloud-key.json

# Define the command to run your FastAPI application using Uvicorn.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

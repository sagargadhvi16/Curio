# Use a slim Python 3.9 image as the base for a smaller footprint
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Install general build essentials and git
# build-essential is for compiling any Python packages with C extensions if needed
# git is for if any requirements are pulled directly from git repos
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create the Backend2 directory within /app where your app code will reside
RUN mkdir /app/Backend2

# Copy requirements.txt from your local Backend2/ to /app/Backend2/ in the container
# The build context is the root of your project (Curio), so `Backend2/` is relative to that.
COPY Backend2/requirements.txt /app/Backend2/requirements.txt

# Install Python dependencies from Backend2/requirements.txt
# --no-cache-dir reduces the size of the Docker image
RUN pip install --no-cache-dir -r /app/Backend2/requirements.txt

# Copy all your application Python files and initial chat_history.json from Backend2/
# to /app/Backend2/ in the container.
COPY Backend2/api.py /app/Backend2/
COPY Backend2/voice_assistant_core.py /app/Backend2/
COPY Backend2/interest_analysis.py /app/Backend2/
COPY Backend2/chat_history.json /app/Backend2/

# Expose the port that FastAPI will run on
EXPOSE 8000

# Define build arguments for GCP Project ID and Location.
# These values will be passed from your docker-compose.yml during the build process.
ARG GCP_PROJECT_ID
ARG GCP_LOCATION

# Set environment variables inside the container for the FastAPI app to use.
# vertexai.init() in voice_assistant_core.py will read these.
ENV GCP_PROJECT_ID=${GCP_PROJECT_ID}
ENV GCP_LOCATION=${GCP_LOCATION}

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable.
# This assumes your 'google-cloud-key.json' file is in the root of your project
# (same directory as docker-compose.yml and Dockerfile)
# It's copied into /app/google-cloud-key.json inside the container.
COPY google-cloud-key.json /app/google-cloud-key.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/google-cloud-key.json

# Define the command to run your FastAPI application using Uvicorn.
# The app is located at /app/Backend2/api.py within the container.
CMD ["uvicorn", "Backend2.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

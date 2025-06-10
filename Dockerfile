FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Install general build essentials if any Python packages need compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the working directory first.
# It's now copied from the build context (which is Backend2) directly into /app
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create the directory for chat history.
# This ensures the directory exists before the application tries to write to it.
# The `voice_assistant_core.py` expects `./Backend2/chat_history.json`.
# So inside the container, we need to recreate the `Backend2` structure relative to `/app`.
# Let's adjust this path to be consistent with how it will be accessed by the volume mount later.
# If `CHAT_HISTORY_FILE = "./Backend2/chat_history.json"` in your Python code,
# and your `docker-compose.yml` maps `backend_chat_history` to `/app/Backend2/chat_history.json`,
# then the Python code *should* find it at the correct path if the volume mount is a file-to-file mount.
# However, `mkdir -p Backend2/database/chat_sessions` creates a directory structure.
# Let's align it with `CHAT_HISTORY_FILE` for clarity.

# Since voice_assistant_core.py uses CHAT_HISTORY_FILE = "./Backend2/chat_history.json",
# and the Dockerfile's WORKDIR is /app, we need to ensure /app/Backend2/ exists.
# We'll copy Backend2's content into /app directly.
# The best practice is to make WORKDIR the root of your source, and then COPY everything.
# Let's simplify the WORKDIR for this.

# Revised strategy:
# 1. Set WORKDIR to /app
# 2. Copy the entire Backend2 directory content into /app/Backend2
# 3. Adjust `CHAT_HISTORY_FILE` in Python code to be relative to the copied structure, or
#    ensure the volume mount directly maps to the file.
#
# Simpler: Make WORKDIR /app/Backend2
# Then all your COPY commands are relative to Backend2

# Let's assume your Dockerfile is in Backend2/ and you want to copy all Backend2 files.
# The context is Backend2 already.

# Corrected COPY commands relative to the `Backend2` context:
# Copy all files from the current build context (Backend2) into the /app directory in the container
# This means /app will contain api.py, voice_assistant_core.py, interest_analysis.py, requirements.txt etc.
COPY . .

# Now, create the directory for chat history.
# This aligns with the path in voice_assistant_core.py: CHAT_HISTORY_FILE = "./Backend2/chat_history.json"
# We need to make sure `Backend2` directory exists relative to `/app` for the volume mount.
# Since we copied everything from `Backend2` into `/app`, the Python code's path
# `./Backend2/chat_history.json` would be looking for `/app/Backend2/chat_history.json`
# It's better to keep the Python code's path simple (e.g., `chat_history.json`)
# if the Dockerfile structure is copying `Backend2`'s contents directly to `/app`.

# Let's make this explicit. In voice_assistant_core.py, update CHAT_HISTORY_FILE to just "chat_history.json"
# and in Dockerfile, create `database` directory directly in `/app`.

# --- REVISION: Simplified chat history path for Docker consistency ---
# This requires a small change in voice_assistant_core.py:
# Change: CHAT_HISTORY_FILE = "./Backend2/chat_history.json"
# To:     CHAT_HISTORY_FILE = "chat_history.json" # Or adjust volume mount more granularly

# Given the `docker-compose.yml` maps `backend_chat_history` to `/app/Backend2/chat_history.json`
# and the `Dockerfile`'s WORKDIR is `/app`, we need the files from `Backend2` to be at `/app/Backend2`.

# To achieve this:
# 1. Keep `docker-compose.yml`'s `context: ./Backend2` and `dockerfile: Dockerfile`.
# 2. Change `Dockerfile` WORKDIR to `/app` (which is already done).
# 3. Change `Dockerfile` `COPY` commands to `COPY . /app/Backend2` or copy specific files where needed.

# Let's fix the COPY command directly.
# Since context is Backend2, and WORKDIR is /app, we copy all *contents* of Backend2
# (which is the context root) directly into /app.
# This means /app will contain api.py, voice_assistant_core.py, etc.
# So, Python's CHAT_HISTORY_FILE should point to `chat_history.json` directly within /app.

# IMPORTANT:
# Please make this minor change in your voice_assistant_core.py (and api.py if it references it):
# CHAT_HISTORY_FILE = "./Backend2/chat_history.json"
# TO:
# CHAT_HISTORY_FILE = "chat_history.json"

# Assuming you've made that change, here's the corrected Dockerfile:

# Copy your Python application files from the build context (which is Backend2)
# into the /app directory inside the container.
# So, /app will contain api.py, voice_assistant_core.py, interest_analysis.py, requirements.txt, chat_history.json
COPY . /app/

# Now, create the directory that your Python code expects for chat history.
# Since chat_history.json is directly in /app, its parent directory is /app.
# The original structure was Backend2/database/chat_sessions.
# If `interest_analysis.py` still looks for `database/chat_sessions`, we need to recreate that.
# Let's simplify and assume chat_history.json directly in /app as well.

# Re-simplifying for the volume mount directly to chat_history.json
# `docker-compose.yml` volume: `- backend_chat_history:/app/Backend2/chat_history.json`
# This means inside the container, the path `/app/Backend2/chat_history.json`
# will be managed by the volume. So your Python code needs to look for it there.

# To align this, we should copy Backend2's content into /app/Backend2.
# OR, change WORKDIR to /app/Backend2 and copy '.' there.

# Let's use the simplest and most robust approach:
# 1. Set WORKDIR /app
# 2. Copy the entire Backend2 directory into /app/Backend2
# This makes internal paths consistent.

# Start fresh with corrected logic:
# Use a slim Python 3.9 image as the base for a smaller footprint
FROM python:3.9-slim-buster

# Set the working directory inside the container to /app
WORKDIR /app

# Install general build essentials if any Python packages need compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create the Backend2 directory within /app
RUN mkdir Backend2

# Copy the requirements.txt from your local Backend2 into /app/Backend2
# The build context is your root project folder (Curio), not Backend2.
# So `COPY Backend2/requirements.txt` means from root.
COPY Backend2/requirements.txt /app/Backend2/requirements.txt

# Install Python dependencies from Backend2/requirements.txt
RUN pip install --no-cache-dir -r /app/Backend2/requirements.txt

# Copy all other Backend2 files into /app/Backend2
COPY Backend2/api.py /app/Backend2/
COPY Backend2/voice_assistant_core.py /app/Backend2/
COPY Backend2/interest_analysis.py /app/Backend2/
COPY Backend2/chat_history.json /app/Backend2/ # Copy the initial chat history template

# Ensure the directory structure for chat_history.json parent is correct if it changes
# voice_assistant_core.py needs os.makedirs(os.path.dirname(file_path))
# With the volume mount, Docker will ensure the parent directory for the *file* exists.
# We don't need `mkdir -p Backend2/database/chat_sessions` now.

# Expose the port that FastAPI will run on.
EXPOSE 8000

# Define build arguments for GCP Project ID and Location.
ARG GCP_PROJECT_ID
ARG GCP_LOCATION

# Set environment variables inside the container for FastAPI app to use.
ENV GCP_PROJECT_ID=${GCP_PROJECT_ID}
ENV GCP_LOCATION=${GCP_LOCATION}

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable.
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/google-cloud-key.json

# Define the command to run your FastAPI application using Uvicorn.
# The app is located at /app/Backend2/api.py
CMD ["uvicorn", "Backend2.api:app", "--host", "0.0.0.0", "--port", "8000"]

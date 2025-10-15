# Stage 1: Base Image and System Setup
# Use a modern, slim Python image for efficiency.
FROM python:3.10-slim

# Set the default application port for Hugging Face Spaces.
ARG APP_PORT=7860
ENV PORT=${APP_PORT}

# Install git, which is a critical dependency for the GitPython library in your project.
# We also include gcc and g++ as they are sometimes needed to compile Python packages.
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Stage 2: User Setup and Environment Security
# Create a non-root user 'user' for better security, which is best practice.
RUN useradd -m -u 1000 user
USER user

# Set the user's home directory and add it to the system path.
ENV HOME=/home/user
ENV PATH="${HOME}/.local/bin:${PATH}"

# Set the working directory inside the container.
WORKDIR /app

# Stage 3: Python Dependencies
# Copy the requirements file first to take advantage of Docker's layer caching.
# This avoids reinstalling dependencies every time you change your application code.
COPY --chown=user requirements.txt .

# Install the Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Stage 4: Application Code and Startup
# Copy all your application files into the container.
COPY --chown=user . .

# Expose the port so the container can receive traffic.
EXPOSE ${APP_PORT}

# Define the command to start your FastAPI application.
# It uses uvicorn, binds to all network interfaces (0.0.0.0), and runs on the port from environment
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]

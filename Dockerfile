# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and install uv -> this is a package manager for python in rust
# https://docs.astral.sh/uv/getting-started/
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Copy application code and requirements
COPY requirements.txt .

# uv install the requirements way faster than pip
RUN uv pip install --no-cache-dir -r requirements.txt --system

COPY . /app

# pull nltk to parse classes from moondream response
RUN python3 setup_nltk.py

EXPOSE 5000

# Set default command
CMD ["python3", "run.py"]
# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set up NLTK data
RUN python3 -m nltk.downloader punkt averaged_perceptron_tagger averaged_perceptron_tagger_eng

# Set default command
CMD ["python3", "run.py"] 
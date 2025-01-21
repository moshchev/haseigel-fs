# haseUNDigel Project

Below you'll find a comprehensive README describing how the haseUNDigel project is structured, what it does, and how to run it—both using a standard Python environment and via Docker. This README assumes you've cloned or downloaded this repository, which contains all the code, configuration, and example files needed.

## Table of Contents
1. Project Overview
2. Main Features
3. Folder Structure
4. Installation & Setup (Non-Docker)
5. Docker Setup & Usage
6. Environment Variables
7. How to Run
8. Usage & API Endpoints
9. Additional Directories
10. Workflow Summary
11. Additional Notes

## Project Overview

haseUNDigel is designed to:
1. Receive HTML data (via API requests or by loading directly from a database).
2. Parse the HTML to identify and extract image URLs.
3. Download images from these URLs into a local temporary folder.
4. Classify the downloaded images using multiple integrated computer vision models:
   - MobileViT (MobileViTV2ForImageClassification)
   - Moondream (a Vision-Language model supporting multiple queries)
   - Async Vision-Language classifiers (using litellm for certain hosted models)

The project is built on Flask for its API and uses Pydantic, NLTK, and Transformers for data validation, text parsing, and model inference. It also supports a PostgreSQL database integration for loading HTML data in bulk, though you can skip the database part and just send HTML directly.

## Main Features

- **HTML Processing**
  - Extract <img> tags and relevant attributes (like src, alt, etc.).
- **Image Downloading**
  - Resolve relative paths, handle SSL vs. non-SSL, check file sizes, and store images in a local temp directory.
- **Image Classification**
  - MobileViTClassifier: Lightweight image classifier for quick predictions.
  - MoondreamProcessor: Asynchronous image encoding & question answering (multiple categories).
  - VisionLanguageModelClassifier: Example of hooking up a hosted LLM service for images.
- **Batch Processing**
  - Concurrency and async flows to handle many images and domains at once.
- **REST API Endpoints**
  - For classification, domain-level HTML processing, single HTML processing, and health checks.

## Folder Structure

Below is a high-level overview of the repository. 

```
haseigel-fs/
├── app/
│   ├── __init__.py              # Initializes the Flask app & registers routes
│   ├── api/
│   │   └── routes.py            # Defines the Flask API endpoints
│   ├── config/
│   │   ├── config.py            # Basic config (directory paths, etc.)
│   │   ├── models.py            # Dictionary mapping model names to model classes
│   │   ├── constants.py         # Constants like error messages, default sizes
│   │   └── __init__.py
│   ├── core/
│   │   ├── image_models.py      # Classes for MobileViT, Moondream, etc.
│   │   ├── response_validation.py # Prompt building & response parsing logic
│   │   └── __init__.py
│   ├── loaders/
│   │   ├── loaders.py           # Classes for loading images in batches
│   │   └── __init__.py
│   ├── services/
│   │   ├── single_image_classification.py
│   │   ├── extract_images.py
│   │   ├── process_domains_moondream.py
│   │   ├── process_domains_hosted.py
│   │   ├── processing_functions.py
│   │   └── __pycache__/
│   ├── utils/
│   │   ├── image_preprocessing.py
│   │   ├── data_tool.py
│   │   └── __init__.py
│   └── __pycache__/
├── data/
│   └── images/
│       └── temp/                # Downloaded images go here
├── docs/
│   └── moondream_performance.md # Performance tests documentation
├── playground/                  # Various scripts and experiments
├── run.py                      # Entry point to run Flask
├── Dockerfile                  # Docker build instructions
├── docker-compose.yml          # Docker Compose configuration
├── requirements.txt            # Dependencies list
├── setup_nltk.py              # Script to download NLTK models
├── README.md                  # This file
```

### Important Files & Directories
- `app/`: Main application logic (API routes, configs, model code, services).
- `data/`: Store downloaded images (under images/temp).
- `docs/`: Documentation or notes (for example, performance notes on Moondream).
- `playground/`: A "lab" folder for personal tests, sample scripts, or experimental code.
- `Dockerfile` & `docker-compose.yml`: Docker setup to containerize the application.
- `requirements.txt`: Python dependencies needed to run the application.
- `setup_nltk.py`: Script for downloading NLTK models you'll need.

## Installation & Setup (Non-Docker)

If you prefer running everything locally without Docker:

1. **Clone the Repository**
```bash
git clone <REPO_URL>  # your company repo link
cd haseigel-fs
```

2. **Create and Activate a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# or
venv\Scripts\activate          # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Set Up Environment Variables**
   - Create a .env file in the project root if needed (see Environment Variables).

5. **Download NLTK Dependencies**
```bash
python setup_nltk.py
```

## Docker Setup & Usage

If you want a Docker-based setup, we provide both a Dockerfile and a docker-compose.yml.

### Prerequisites:
- Docker installed and running
- (Optional) Docker Compose for multi-container orchestration

### Build and Run with Docker:

#### a. Using plain Docker
```bash
# Build the image
docker build -t haseundigel-app .

# Run the container
docker run --rm -p 5000:5000 haseundigel-app
```

This will:
- Install system dependencies (Python 3.10, etc.).
- Install Python packages from requirements.txt.
- Expose Flask on port 5000.

#### b. Using Docker Compose
```bash
docker-compose up --build
```

### GPU / NVIDIA Support:
- The provided Dockerfile uses nvidia/cuda:12.3.1-runtime-ubuntu22.04.
- The docker-compose.yml includes reservations for GPU capabilities.
- Make sure you have the NVIDIA Container Toolkit installed for GPU support.

### Confirm:
Once the container is running, open http://localhost:5000/health. You should get:
```json
{"status": "ok"}
```

## Environment Variables

The app uses python-dotenv to load environment variables from a .env file.

Common environment variables include:
- **OpenAI & litellm:**
  - `OPENAI_API_KEY` (or other keys if you use a hosted LLM provider).
- **Database** (if you want to store/fetch HTML from Postgres):
  - `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASS`, `DB_PORT`.
- Additional keys or tokens for Fireworks AI models, etc.

Example `.env`:
```env
OPENAI_API_KEY="your_openai_api_key_here"
DB_HOST="localhost"
DB_NAME="mydatabase"
DB_USER="postgres"
DB_PASS="mysecretpassword"
DB_PORT=5432
```

## How to Run

### 1. Locally (Python / Flask)

After installing dependencies and setting environment variables:

```bash
# Optionally set FLASK_APP (for older Flask versions)
export FLASK_APP=app

# Then run
flask run
```

Or simply:
```bash
python run.py
```

### 2. Via Docker / Docker Compose

- **Plain Docker:**
```bash
docker build -t haseundigel-app .
docker run --rm -p 5000:5000 haseundigel-app
```

- **Docker Compose:**
```bash
docker-compose up --build
```

Then visit http://127.0.0.1:5000/health to confirm.

## Usage & API Endpoints

### 1. Health Check
`GET /health`
Returns `{ "status": "ok" }` if the server is running.

### 2. Image Classification
`POST /model/<model_name>`
- `<model_name>` can be one of: mobilevit_v2, moondream, vllm
- Send an image file as form-data (key="image").

Example:
```bash
curl -X POST -F "image=@/path/to/image.jpg" \
     http://127.0.0.1:5000/model/mobilevit_v2
```

### 3. Process Domains
`POST /process-domains`
```json
{
  "data": [
    {
      "domain_start_id": 123,
      "base_url": ["http://example.com"],
      "response_text": ["<html>...</html>"]
    }
  ],
  "output_type": "detailed"
}
```

### 4. Process Single HTML
`POST /process-html`
```json
{
  "response_text": "<html>...</html>",
  "response_url": "http://example.com"
}
```

## Additional Directories

### 1. data/
- Contains subfolders like images/temp for downloaded images.

### 2. docs/
- Currently holds moondream_performance.md with benchmarks.

### 3. playground/
- Various test and experimentation scripts.

## Workflow Summary
1. Receive HTML data via POST /process-domains or POST /process-html.
2. Extract <img> tags with extract_images.py (BeautifulSoup).
3. Download images into TEMP_IMAGE_DIR.
4. Classify images with the chosen model.
5. Return aggregated predictions and stats in JSON form.

## Additional Notes

### Database Integration (optional):
- In app/utils/data_tool.py, you'll find methods for PostgreSQL integration.

### Extending the Project:
- To add new classification models, create a new class in app/core/image_models.py
- For new endpoints, add them in app/api/routes.py or create a new blueprint.

### NLTK:
- If you see errors about missing tokenizers, run `python setup_nltk.py`

### GPU Acceleration:
- Dockerfile is based on nvidia/cuda:12.3.1-runtime-ubuntu22.04.
- Make sure you have installed the NVIDIA container toolkit for GPU usage.


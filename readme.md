# Hase&Igel Project

## Table of Contents
1. Project Overview
2. Main Features
3. Folder Structure
4. Installation & Setup (Non-Docker)
5. Docker Setup & Usage
6. Environment Variables
7. Usage & API Endpoints
8. Additional Directories
9. Workflow Summary
10. Additional Notes

## Project Overview

haseUNDigel is designed to:
1. Receive HTML data via provided endpoints. 
2. Parse the HTML to identify and extract image URLs.
3. Download images from these URLs into a local temporary folder.
4. Classify the downloaded images using multiple integrated computer vision models:
   - MobileViT (MobileViTV2ForImageClassification) # you can use it for tests of the enpoints.
   - Moondream (a Vision-Language model supporting multiple queries)
   - Async Vision-Language classifiers (using litellm to access any hosted models)

The project is built on Flask for its API and uses Pydantic, NLTK, and Transformers for data validation, text parsing, and model inference.

## Main Features

- **HTML Processing**
  - Extract image tags and relevant attributes (like src, alt, etc.).
- **Image Downloading**
  - Resolve relative paths, handle SSL vs. non-SSL, check file sizes, and store images in a local temp directory.
- **Image Classification** # we left 3 main classifiers inside 
  - MobileViTClassifier: Lightweight image classifier for quick predictions.
  - MoondreamProcessor: Asynchronous image encoding & question answering (multiple categories) & description with custom categories
  - VisionLanguageModelClassifier: LiteLLM interface that can access most of the hosted LLMs.
- **Batch Processing**
  - Concurrency and async flows to handle many images and domains at once.

## Folder Structure

Below is a high-level overview of the repository. 

```
haseigel-fs/
├── app/
│   ├── __init__.py              # Initializes the Flask app & registers routes
│   ├── api/
│   │   └── routes.py            # Defines the Flask API endpoints
│   ├── data/
│   └── images/
│   │   └── temp/                # Downloaded images go here
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
├── docs/
│   ├── moondream_performance.md                    # Performance tests documentation
│   ├── hosted_model_implementation.md              # Infos on the implementation of litellm
│   └── moondream_implementation.md                 # Moondream implemenation 
├── playground/                  # Various scripts and experiments
├── run.py                      # Entry point to run Flask
├── Dockerfile                  # Docker build instructions
├── docker-compose.yml          # Docker Compose configuration
├── requirements.txt            # Dependencies list
├── setup_nltk.py              # Script to download NLTK models
├── README.md                  # This file
```

### Important Files & Directories

- `app/`: Main application logic (API routes, configs, model code, services, data folder).
- `app/data/`: Stores test images (under images/temp).
- `docs/`: Documentation or notes (for example, performance notes on Moondream).
- `playground/`: A "lab" folder for personal tests, sample scripts, or experimental code.
- `Dockerfile` & `docker-compose.yml`: Docker setup to containerize the application.
- `requirements.txt`: Python dependencies needed to run the application.
- `setup_nltk.py`: Script for downloading NLTK models you'll need.

## Installation & Setup (Non-Docker)

If you prefer running everything locally without Docker:

1. **Clone the Repository**

```bash
git clone https://github.com/moshchev/haseigel-fs.git  # your company repo link
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

6. **Run the Service**
```bash
python run.py
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

# Run the container witout gpu support
docker run --rm -p 5000:5000 --name image-html-parser haseundigel-app

# Run the container with gpu support
docker run --gpus all --rm -p 5000:5000 --name image-html-parser haseundigel-app
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

Once the container is running, send the get request to http://localhost:5000/health. You should get:

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

https://docs.litellm.ai/docs/providers - you can find all the providers that are currently supported by litellm, so you can configure your .env file to use the provider you want.

## Usage & API Endpoints

### 1. Health Check
`GET /health`
Returns `{ "status": "ok" }` if the server is running.

### 2. Image Classification
`POST /model/<model_name>`
- `<model_name>` can be one of: mobilevit_v2, moondream, any model that is supported by litellm
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

we have a tool in utils that fetches the data from the database and prepares it to the right format
app.utils.data_tool.py - get_html_data_as_json()


### 4. Process Single HTML
`POST /process-html`
```json
{
  "response_text": "<html>...</html>",
  "response_url": "http://example.com"
}
```

## Workflow Summary
1. Receive HTML data via POST /process-domains or POST /process-html.
2. Extract <img> tags with extract_images.py (BeautifulSoup).
3. Download images into TEMP_IMAGE_DIR.
4. Classify images with the chosen model.
5. Return aggregated predictions and stats in JSON form.

## Additional Notes

### Extending the Project:
- To add new classification models, create a new class in app/core/image_models.py
- For new endpoints, add them in app/api/routes.py or create a new blueprint.

### NLTK:
- If you see errors about missing tokenizers, run `python setup_nltk.py`

### GPU Acceleration:
- Dockerfile is based on nvidia/cuda:12.3.1-runtime-ubuntu22.04.
- Make sure you have installed the NVIDIA container toolkit for GPU usage. Pass the --gpus all flag to the docker run command.

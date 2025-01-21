# Table of Contents
1. Project Overview
2. Main Features
3. Folder & File Structure
4. Installation & Setup
5. Environment Variables
6. How to Run
7. Usage & API Endpoints
8. Workflow Summary
9. Additional Notes

## Project Overview

The Project is designed to:
1. Receive HTML data (via API requests or by loading directly from a database).
2. Parse the HTML to identify and extract image URLs.
3. Download images from these URLs into a local temporary folder.
4. Classify images using one of several integrated computer vision models:
   - MobileViT (MobileViTV2ForImageClassification)
   - Moondream (a Vision-Language model supporting multiple queries)
   - Async Vision-Language classifiers (using litellm for certain hosted models)

The project is built with Flask for serving an API layer and uses Pydantic, NLTK, and Transformers for data validation, image classification, and text extraction.
Additionally, it supports a database integration (PostgreSQL) for loading/storing HTML data and can also handle domain-based HTML input in batches.

## Main Features
- HTML Processing: Extract <img> tags and relevant attributes (src, alt, etc.)
- Image Downloading: Resolves relative paths, handles SSL verification or non-verification, and applies basic file size checks
- Image Classification:
  - MobileViTClassifier: Lightweight image classifier for quick predictions
  - MoondreamProcessor: Asynchronous image encoding & question answering for more complex tasks or multiple categories
  - VisionLanguageModelClassifier: Example of using a hosted LLM for image-based queries
- Batch Processing: Use concurrency and async flows to handle multiple images and domains efficiently
- REST API Endpoints: Endpoints for model classification, HTML processing, domain-level processing, and health checks

## Folder & File Structure

Below is a high-level view of the app folder (named `app/`). Most of your development and customization will happen in these directories:

```
app/
├── __init__.py               # Initializes the Flask app & registers routes
├── api/
│   └── routes.py             # Defines the Flask API endpoints
├── config/
│   ├── config.py             # Basic config (paths, etc.)
│   ├── models.py             # References to different model classes
│   ├── constants.py          # Constants like error messages, image size
│   └── __init__.py
├── core/
│   ├── image_models.py       # Classes for MobileViT, MoondreamProcessor, VisionLanguageModelClassifier, etc.
│   ├── response_validation.py# Prompt building & response parsing logic
│   └── __init__.py
├── loaders/
│   ├── loaders.py            # Classes that load images + model loader logic
│   └── __init__.py
├── services/
│   ├── single_image_classification.py  # Logic for classifying a single image with a chosen model
│   ├── extract_images.py      # Functions to parse HTML & download images
│   ├── process_domains_moondream.py # Asynchronous domain-based processing example
│   ├── processing_functions.py# Functions to process HTML, domains, orchestrate classification
│   └── __pycache__/
├── utils/
│   ├── image_preprocessing.py # Base64 encoding & resizing (currently used by "prepare_image")
│   ├── data_tool.py           # DB loading, random HTML retrieval, etc.
│   └── __init__.py
├── __pycache__/
└── data/                     # (Dynamically created) or used for storing images / HTML data
```

### Highlights of Each Subfolder
- **api/**: Contains the Flask blueprint (routes.py) with endpoints to interact with the services
- **config/**: Configuration files (e.g., model registry, global constants, and directory setups)
- **core/**: Core classes for classification and prompt/response logic, including the different model wrappers
- **loaders/**: Classes to load images in batches, handle threading, or coordinate model loading
- **services/**: Contains the main "business logic," like HTML parsing, downloading images, classifying them, and providing domain-level logic
- **utils/**: Helper functions for image preprocessing (resizing, base64 encoding) and database interactions

## Installation & Setup
1. Clone the Repository
```bash
git clone <REPO_URL>  # your company repo link
cd haseigel-fs  # or the name of the folder
```

2. Create and Activate a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # on macOS/Linux
# or 
venv\Scripts\activate      # on Windows
```

3. Install Dependencies
Make sure you have a requirements.txt or Pipfile (not shown here, but typically included). Then run:
```bash
pip install -r requirements.txt
```
(Adjust if your project uses Poetry, pipenv, or another dependency manager.)

4. Set Up Environment Variables
For a quick start, create a `.env` file in the project root (same level as your `app/` folder) and define the variables needed. See Environment Variables for details.

## Environment Variables

The project uses python-dotenv to load environment variables from a `.env` file. Key variables include:
- OpenAI & litellm:
  - `OPENAI_API_KEY` (or other keys if you're using hosted LLM providers)
- PostgreSQL Database:
  - `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASS`, `DB_PORT`
- Others may be required depending on your environment (e.g., for Fireworks AI or additional model providers)

A minimal example `.env`:
```env
OPENAI_API_KEY=your_openai_api_key
DB_HOST=localhost
DB_NAME=mydatabase
DB_USER=postgres
DB_PASS=mysecretpassword
DB_PORT=5432
```

## How to Run
1. Flask Application
Make sure you're in the project's root directory, where your `app/` folder resides.
```bash
export FLASK_APP=app   # (if needed, for older Flask versions)
flask run
```
By default, it starts on http://127.0.0.1:5000.

2. Confirm Health Check
Open your browser or use curl to check the `/health` endpoint:
```bash
curl http://127.0.0.1:5000/health
```
If everything is running, you should see a JSON response: `{"status": "ok"}`

## Usage & API Endpoints
1. Health Check
   - `GET /health`
   - Returns a simple `{ "status": "ok" }` to confirm the server is running

2. Image Classification
   - `POST /model/<model_name>`
   - Upload an image file with form-data under the key "image"
   - `<model_name>` can be one of: `mobilevit_v2`, `moondream`, or `vllm` (as defined in MODEL_CLASSES)

Example using curl:
```bash
curl -X POST -F "image=@/path/to/your/image.jpg" \
     http://127.0.0.1:5000/model/mobilevit_v2
```
Returns JSON with the classification result.

3. Process Domains
   - `POST /process-domains`
   - Expects a JSON body of the form:
```json
{
  "data": [
    {
      "domain_start_id": 123,
      "base_url": ["http://example.com"],
      "response_text": ["<html>...</html>"]
    },
    ...
  ],
  "output_type": "detailed"   // optional
}
```
   - This endpoint parses each domain's HTML, extracts and downloads images, then classifies them using the MobileViTClassifier (by default)
   - The output_type can be "detailed" or "summary"

4. Process Single HTML
   - `POST /process-html`
   - Expects a JSON body like:
```json
{
  "response_text": "<html>...</html>",
  "response_url": "http://example.com"
}
```
   - Similar to `/process-domains`, but processes only one HTML snippet

## Workflow Summary
1. Receive JSON with HTML content:
   - The user calls `/process-domains` or `/process-html` with the relevant HTML data (and base URLs)
2. Extract Image Tags:
   - The code (`extract_images.py`) uses BeautifulSoup to find all `<img>` tags
3. Download Images:
   - Valid image URLs get downloaded into the `TEMP_IMAGE_DIR` (`app/config/config.py` points to `app/data/images/temp`)
4. Classify Images:
   - Once downloaded, each image is passed to the requested classification model (e.g., MobileViT)
   - The classification results are compiled into a JSON response
5. Return Results:
   - The API endpoint returns aggregated predictions plus any relevant counts/statistics about the categories

## Additional Notes
### Database Integration
The project can optionally fetch HTML data from a PostgreSQL database using `app.utils.data_tool`. For local usage or simple testing, you don't necessarily need the DB portion; you can directly send the HTML in requests.

### Async Models
For advanced usage (like using MoondreamProcessor or AsyncVisionLanguageModelClassifier), check out the examples in the `services/` folder (e.g., `process_domains_moondream.py`) to run asynchronous classification pipelines.

### Extending the Project
- If you want to integrate new classification models, simply create a new class following the pattern in `core/image_models.py` and reference it in `config/models.py`
- For new endpoints or workflows, add a route to `api/routes.py` and connect it to the relevant service function


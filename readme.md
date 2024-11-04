## How to Start the App

1. **Install Dependencies**:
   - Using `pip`:
     ```sh
     pip install -r requirements.txt
     ```

2. **Run the Flask App**:
   ```sh
    python3 run.py
   ```

## API Endpoints

### `/process-domains` (POST)
- **Description**: Processes domain data and returns the result.
- **Request Body**: JSON object containing domain data and an optional `output_type` (default is `detailed`).
- **Response**: JSON object with the processed domain data.

### `/health` (GET)
- **Description**: Health check endpoint to verify if the app is running.
- **Response**: JSON object with status `ok`.

### `/model/<model_name>` (POST)
- **Description**: Classifies an image using the specified model.
- **Request**: Multipart form-data with an image file.
- **Response**: JSON object with the classification result.

## Available Models

### MobileViTClassifier
- **Model Name**: `mobilevit_v2`
- **Description**: Uses the MobileViTV2 model for image classification.
- **Pretrained Weights**: `shehan97/mobilevitv2-1.0-imagenet1k-256`
- **Prediction Output**: Returns the predicted ImageNet class label for the input image.

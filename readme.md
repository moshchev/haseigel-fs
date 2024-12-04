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
- **Example Input**:
  ```json
  {
    "data": [
      {
        "domain_start_id": 123,
        "response_text": ["<html>...</html>", "<html>...</html>"]
      },
      {
        "domain_start_id": 124,
        "response_text": ["<html>...</html>"]
      }
    ]
  }
  ```

- **Example Output**:
  ```json
  {
    "status": "success", 
    "output": {
      "details": [
        {
          "domain_start_id": 123,
          "statistics": {
            "cats": 10,
            "dogs": 5,
            "logos": 3,
            "other": 2
          },
          "total_images": 20
        },
        {
          "domain_start_id": 124,
          "statistics": {
            "cats": 5,
            "dogs": 10,
            "logos": 2,
            "other": 1
          },
          "total_images": 18
        }
      ],
      "summary": {
        "total_domains": 2,
        "total_images": 38,
        "aggregated_statistics": {
          "cats": 15,
          "dogs": 15,
          "logos": 5,
          "other": 3
        }
      }
    }
  }
  ```


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

### OpenAI Vision Model
- **Model Name**: `openai_vision`
- **Description**: Uses OpenAI's GPT-4 (o and o-mini) Vision model for image analysis and classification.
- **API Integration**: Requires valid OpenAI API key configured in environment variables.
- **Prediction Output**: Returns predicted class label in the same format as the MobileViTClassifier.


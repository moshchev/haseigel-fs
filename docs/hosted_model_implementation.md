# Hosted Vision-Language Model Implementation Details

The `AsyncVisionLanguageModelClassifier` class provides an asynchronous interface for processing images using hosted vision-language models like GPT-4V and Qwen-VL. Here's a detailed breakdown of its implementation:

## Core Components

### Initialization

- Configurable model selection from supported hosted models (GPT-4V, Llama-V3, Qwen-VL etc.)
- Uses litellm for unified API access to different model providers -> https://docs.litellm.ai/docs/providers
- System prompt configuration for consistent model behavior

### Key Methods

#### Message Preparation

- `_prepare_message()`: Converts image to base64 and constructs API-compatible message format
- `prepare_batch_messages()`: Handles batch preparation of multiple images asynchronously
- Supports custom prompts based on categorization needs

#### Prediction Methods

1. `predict()`
   - Processes single image with optional categories
   - Returns raw model response text
   - Example:
   ```python
   response = await predict(image_path, categories=['cat', 'dog'])
   ```

2. `predict_batch()`
   - Two-stage batch processing:
     1. Message preparation in large batches (prep_batch_size)
     2. API requests in smaller chunks (request_batch_size)
   - Returns list of predictions mapped to file paths
   - Example:
   ```python
   results = await predict_batch(
       image_paths,
       categories=['cat', 'dog'],
       prep_batch_size=20,
       request_batch_size=2
   )
   ```

### Helper Functions

- `clean_llm_output()`: Sanitizes and parses JSON responses from models # You can have this as inspiration. You need to at least catch errors
- Handles removal of markdown code blocks and whitespace
- Converts cleaned text to dictionary

## Performance Considerations

- Asynchronous processing for improved throughput
- Configurable batch sizes for memory and API rate limit management
- Two-stage batching to balance memory usage and API efficiency
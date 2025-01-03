# Moondream2B Performance Comparison

This document compares the performance of Moondream2B model using two different implementations:

1. HuggingFace client
2. Native Moondream client

## Test Environment

### Hardware Specifications

- **Instance Type:** m7i-flex.xlarge (CPU only)
- **CPU:** 4 vCPUs
- **Memory:** 16.0 GiB
- **Network Bandwidth:** 12.5 Gbps

- **Instance Type:** g4dn.xlarge (GPU)
- **GPU:** Nvidia T4
- **VRAM:** 16.0 GiB

## Performance Results

### HuggingFace Implementation
*CPU*

| Operation | Time (seconds) |
|-----------|---------------|
| Model Loading | 5.44 |
| Image Processing | 8.21 |
| Query 1 | 7.77 |
| Query 2 | 7.51 |
| Query 3 | 7.00 |
| **Total Time** | **35.93** |

*GPU*

| Operation | Time (seconds) |
|-----------|---------------|
| Model loaded | 5.60 |
| Image processed | 0.72 |
| Query 1 answered | 1.50 |
| Query 2 answered | 0.52 |
| Query 3 answered | 0.53 |
| **Total Time** | **8.87** |


### Native Moondream Implementation
*Supports only CPU*

| Operation | Time (seconds) |
|-----------|---------------|
| Model Loading | 6.51 |
| Image Processing | 7.15 |
| Query 1 | 0.55 |
| Query 2 | 0.48 |
| Query 3 | 0.49 |
| **Total Time** | **15.18** |

## Key Findings

- Native client is faster overall, with better image processing performance
- Native client supports CPU only -> hf can be faster with GPU, didnt test it yet
- HF doesnt takes shit ton of time to process individual queries, where native client takes 0.48-0.55 seconds per query (kinda ridiculous, I will double check if this is correct)

- GPU is almost twice as fast as best CPU implementation (on 3 queries)
    - Also the model will loaded in the app, so this time will be eliminated in the future. The GPU will take ~ 3 seconds to encode the image *and* answer 3 queries
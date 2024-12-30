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

## Performance Results

### HuggingFace Implementation

| Operation | Time (seconds) |
|-----------|---------------|
| Model Loading | 3.06 |
| Image Processing | 14.80 |
| **Total Time** | **17.86** |

### Native Moondream Implementation

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
- *I didnt test the hf on the multiple queries, still needs to be tested*
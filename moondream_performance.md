I've tested the moondream2b model with two different clients. In the first case I downloaded model from huggingface and in the second case I manuall pulled weights and used the native client from moondream.
Native client is a bit faster than the huggingface client, but supports only cpu. 

specs:
m7i-flex.xlarge -> cpu only
cpu: 4 vCPUs
memory: 16.0 GiB
bandwidth: 12.5 Gibps

model - moondream2gb

Model loaded via huggingface
Model loaded in 3.06 seconds
Image processed in 14.80 seconds
Total time: 17.86 seconds

Native client from moondream
Model loaded in 6.51 seconds
Image processed in 7.15 seconds
Query 1 answered in 0.55 seconds
Query 2 answered in 0.48 seconds
Query 3 answered in 0.49 seconds

Total execution time: 15.18 seconds
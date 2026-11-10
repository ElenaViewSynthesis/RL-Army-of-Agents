| Task | Solution | Reward reason | Solution µs | Eager µs | torch.compile µs | vs eager |
|---|---|---|---:|---:|---:|---|
| Candidate top-k | CUDA | correct | 604,808.9 | 4,274.4 | 3,337.5 | 141.5× slower |
| Candidate top-k | Triton | correct | 9,008.4 | 4,271.4 | 3,338.1 | 2.1× slower |
| Tree step | CUDA | correct | 11,610.3 | 171.3 | 18.9 | 67.8× slower |
| Tree step | Triton | faster_than_eager | 98.7 | 176.9 | 21.5 | 1.8× faster |
| Row attention (32k) | CUDA | correct | 224,276.4 | 30,757.7 | 11,034.5 | 7.3× slower |
| Row attention (32k) | Triton | correct | 134,464.0 | 30,754.0 | 11,082.9 | 4.4× slower |
| Verify attention (32k) | CUDA | correct | 1,703,356.2 | 267,099.8 | 97,101.1 | 6.4× slower |
| Verify attention (32k) | Triton | correct | 1,035,069.3 | 267,080.3 | 97,000.9 | 3.9× slower |
| GDN tree conv | CUDA | faster_than_both | 7.8 | 172.0 | 9.1 | 22.1× faster |
| GDN tree conv | Triton | faster_than_both | 7.4 | 173.9 | 9.3 | 23.5× faster |
| Mamba2 tree conv | CUDA | faster_than_eager | 26.1 | 185.0 | 8.1 | 7.1× faster |
| Mamba2 tree conv | Triton | faster_than_eager | 25.9 | 184.1 | 8.1 | 7.1× faster |
| GDN recurrence | CUDA | faster_than_both | 1,611.3 | 5,173.1 | 5,394.0 | 3.2× faster |
| GDN recurrence | Triton | faster_than_both | 449.6 | 5,171.8 | 5,445.1 | 11.5× faster |
| GDN recurrence | Triton chain replay | faster_than_both | 521.2 | 5,191.5 | 5,427.9 | 10.0× faster |
| Greedy acceptance | CUDA | faster_than_both | 9.6 | 1,968.2 | 107.8 | 204.5× faster |
| Greedy acceptance | Triton | faster_than_both | 8.6 | 1,969.7 | 107.7 | 229.8× faster |
| Sampled acceptance | CUDA | correct | 46,588.0 | 21,252.4 | 4,557.4 | 2.2× slower |
| Sampled acceptance | Triton | faster_than_both | 3,318.2 | 21,242.2 | 4,649.2 | 6.4× faster |

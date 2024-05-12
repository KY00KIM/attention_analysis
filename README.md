# Attention Performance Analysis

Performance analysis on [FlashAttention2](https://github.com/Dao-AILab/flash-attention) and [PagedAttention](https://github.com/vllm-project/vllm) compared to baseline as [TorchSDPA-Math](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.SDPAParams) via inference on `llama-2-7b` with various sequence length

- Metrics:
  - Latency : sec
  - MaxMemoryAllocated : Torch profiler
  - MaxMemoryReserved : Torch profiler


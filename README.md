## Running the IMO Bench(Answer Bench) Evaluation
Serve the model via vllm:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 40960 \
  --max-num-seqs 1024 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 10240 \
  --enable-prefix-caching \
  --enable-sleep-mode \
  --logprobs-mode processed_logprobs \
  --disable-custom-all-reduce \
  --gpu-memory-utilization 0.6 \
  --disable-log-stats \
  --tensor-parallel-size 4 \
  --seed 0 \
  --override-generation-config '{"max_new_tokens": 32768}' \
  --hf-overrides '{}' \
  --scheduling-policy fcfs \
  --host 0.0.0.0 \
  --port 8001
```

```bash
JUDGE_TOKEN=your_token_here JUDGE_URL=https://your-judge-endpoint/v1/chat/completions python3 eval.py
```

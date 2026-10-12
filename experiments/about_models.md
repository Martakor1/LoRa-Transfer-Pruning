deepseek-ai/DeepSeek-V2-Lite-Chat: ~ 30GB for weights bf16 without bnb. And 28.6 with it only for weights (cant load in fp32)! But possible to train part of weights. Supports MOE, but can't prune it yet.

meta-llama/Llama-3.1-8B-Instruct: 15 gb for weights bf16 + 15 for train, without MOE

google/gemma-4-E4B-it: 15.251 Gb for weights bf16 (and the same for train), no MOE! MOE version: A4B - can't load weights in 40GB!

Qwen/Qwen3-30B-A3B-GPTQ-Int4 ~ 16 GB weight, quantized (MarlinLinear, GPTQ, float16 native). Problems with torch pruning for quantized model, can't build prune groups. MOE.
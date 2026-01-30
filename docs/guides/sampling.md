# Sampling Guide

This guide covers text generation and inference with TenSafe.

## Basic Sampling

Generate text from your fine-tuned model:

```python
# Single prompt
result = tc.sample("Once upon a time")

print(result.samples[0].completion)
```

## Batch Sampling

Generate completions for multiple prompts:

```python
prompts = [
    "The capital of France is",
    "Machine learning is",
    "In the year 2050,",
]

result = tc.sample(prompts, max_tokens=100)

for sample in result.samples:
    print(f"Prompt: {sample.prompt}")
    print(f"Completion: {sample.completion}")
    print(f"Tokens: {sample.tokens_generated}")
    print(f"Finish reason: {sample.finish_reason}")
    print()
```

## Sampling Parameters

### Temperature

Controls randomness in generation:

```python
# Deterministic (greedy)
result = tc.sample(prompt, temperature=0.0)

# Balanced (default)
result = tc.sample(prompt, temperature=0.7)

# Creative/diverse
result = tc.sample(prompt, temperature=1.2)
```

| Temperature | Effect |
|-------------|--------|
| 0.0 | Deterministic, always picks highest probability |
| 0.1-0.5 | Conservative, focused outputs |
| 0.7-0.9 | Balanced creativity and coherence |
| 1.0+ | More random, diverse outputs |

### Top-p (Nucleus Sampling)

Samples from the smallest set of tokens whose cumulative probability exceeds p:

```python
# Conservative
result = tc.sample(prompt, top_p=0.5)

# Standard (default)
result = tc.sample(prompt, top_p=0.9)

# More diverse
result = tc.sample(prompt, top_p=0.95)
```

### Top-k

Limits sampling to the k most likely tokens:

```python
# Very focused
result = tc.sample(prompt, top_k=10)

# Default
result = tc.sample(prompt, top_k=50)

# Disable top-k (use only top-p)
result = tc.sample(prompt, top_k=0)
```

### Max Tokens

Control output length:

```python
# Short responses
result = tc.sample(prompt, max_tokens=32)

# Standard
result = tc.sample(prompt, max_tokens=128)

# Long form
result = tc.sample(prompt, max_tokens=512)
```

### Stop Sequences

Stop generation at specific strings:

```python
result = tc.sample(
    prompt="Write a haiku:\n",
    max_tokens=100,
    stop_sequences=["\n\n", "---"],
)

# Generation stops when encountering stop sequence
print(result.samples[0].finish_reason)  # "stop" or "length"
```

## Complete Sampling Example

```python
result = tc.sample(
    prompts=["Explain quantum computing:", "Write a poem about AI:"],
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    stop_sequences=["###", "\n\n\n"],
)

for sample in result.samples:
    print(f"=== {sample.prompt} ===")
    print(sample.completion)
    print(f"[{sample.tokens_generated} tokens, {sample.finish_reason}]")
    print()
```

## Sampling Configuration

Pre-configure sampling defaults:

```python
from tg_tinker import SamplingConfig

sampling = SamplingConfig(
    max_tokens=200,
    temperature=0.8,
    top_p=0.9,
    top_k=40,
    stop_sequences=["<|endoftext|>"],
)
```

## Model Step Tracking

Samples are generated from the current model state:

```python
# Sample before training
result1 = tc.sample("Hello")
print(f"Sampled at step: {result1.model_step}")  # 0

# Train the model
for batch in batches:
    tc.forward_backward(batch).result()
    tc.optim_step().result()

# Sample after training
result2 = tc.sample("Hello")
print(f"Sampled at step: {result2.model_step}")  # N
```

## Evaluation Workflows

### Periodic Evaluation

```python
eval_prompts = [
    "Q: What is 2+2?\nA:",
    "Translate to French: Hello",
    "Summarize: ...",
]

for step, batch in enumerate(dataloader):
    tc.forward_backward(batch).result()
    tc.optim_step().result()

    # Evaluate every 100 steps
    if (step + 1) % 100 == 0:
        result = tc.sample(eval_prompts, temperature=0.0)
        for sample in result.samples:
            print(f"Step {step+1}: {sample.completion[:50]}...")
```

### A/B Comparison

Compare outputs at different checkpoints:

```python
# Sample from current model
result_before = tc.sample(eval_prompts, temperature=0.0)

# Continue training
train_steps(tc, 1000)

# Sample again
result_after = tc.sample(eval_prompts, temperature=0.0)

# Compare
for before, after in zip(result_before.samples, result_after.samples):
    print(f"Prompt: {before.prompt}")
    print(f"Before: {before.completion}")
    print(f"After: {after.completion}")
    print()
```

## Best Practices

### 1. Use Low Temperature for Evaluation

```python
# Reproducible evaluation
result = tc.sample(
    eval_prompts,
    temperature=0.0,  # Greedy decoding
)
```

### 2. Handle Long Generations

```python
result = tc.sample(prompt, max_tokens=1024)

if result.samples[0].finish_reason == "length":
    print("Warning: Generation truncated")
```

### 3. Batch for Efficiency

```python
# Efficient: single batch request
result = tc.sample(prompts=all_prompts, max_tokens=100)

# Inefficient: many individual requests
for prompt in all_prompts:
    tc.sample(prompt, max_tokens=100)
```

### 4. Combine Parameters Carefully

```python
# Good combination
result = tc.sample(
    prompt,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
)

# Avoid very high temperature with high top_p
# result = tc.sample(prompt, temperature=1.5, top_p=0.99)  # May produce nonsense
```

## Next Steps

- [Training Guide](training.md) - Training workflows
- [Privacy Guide](privacy.md) - DP and homomorphic encryption
- [API Reference](../api-reference/training-client.md) - Full TrainingClient API

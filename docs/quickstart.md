# Quickstart

This guide walks you through your first fine-tuning job with TenSafe.

## Prerequisites

1. Install the SDK: `pip install tg-tinker`
2. Set your API key: `export TS_API_KEY=ts-your-key`

## Basic Training Loop

```python
from tg_tinker import (
    ServiceClient,
    TrainingConfig,
    LoRAConfig,
    OptimizerConfig,
)

# 1. Initialize the client
service = ServiceClient()

# 2. Configure training
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        learning_rate=1e-4,
        weight_decay=0.01,
    ),
    batch_size=8,
    gradient_accumulation_steps=4,
)

# 3. Create training client
tc = service.create_training_client(config)
print(f"Created training client: {tc.id}")

# 4. Training loop
for batch in dataloader:
    # Queue forward-backward pass
    fb_future = tc.forward_backward({
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    })

    # Queue optimizer step
    opt_future = tc.optim_step()

    # Get results
    fb_result = fb_future.result()
    opt_result = opt_future.result()

    print(f"Step {tc.step}: loss={fb_result.loss:.4f}")

# 5. Save checkpoint
checkpoint = tc.save_state()
print(f"Saved checkpoint: {checkpoint.artifact_id}")
```

## With Differential Privacy

Add DP protection to your training:

```python
from tg_tinker import DPConfig

config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16),
    dp_config=DPConfig(
        enabled=True,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_epsilon=8.0,
        target_delta=1e-5,
    ),
)

tc = service.create_training_client(config)

# Training loop with DP
for batch in dataloader:
    fb_future = tc.forward_backward(batch)
    opt_future = tc.optim_step(apply_dp_noise=True)

    fb_result = fb_future.result()
    opt_result = opt_future.result()

    # Check privacy budget
    if opt_result.dp_metrics:
        print(f"Epsilon spent: {opt_result.dp_metrics.total_epsilon:.4f}")
```

## Text Generation

Sample from your fine-tuned model:

```python
# Generate text
result = tc.sample(
    prompts=["Once upon a time", "The quick brown fox"],
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
)

for sample in result.samples:
    print(f"Prompt: {sample.prompt}")
    print(f"Completion: {sample.completion}")
    print()
```

## Checkpoint Management

Save and restore training state:

```python
# Save with metadata
checkpoint = tc.save_state(
    include_optimizer=True,
    metadata={
        "epoch": 1,
        "dataset": "my-dataset",
        "notes": "Best model so far",
    }
)

print(f"Artifact ID: {checkpoint.artifact_id}")
print(f"Size: {checkpoint.size_bytes} bytes")
print(f"Encryption: {checkpoint.encryption.algorithm}")

# Later: restore from checkpoint
tc.load_state(checkpoint.artifact_id)
print(f"Restored to step: {tc.step}")
```

## Error Handling

Handle common errors gracefully:

```python
from tg_tinker import (
    TGTinkerError,
    RateLimitedError,
    DPBudgetExceededError,
    FutureTimeoutError,
)

try:
    fb_future = tc.forward_backward(batch)
    result = fb_future.result(timeout=60)

except FutureTimeoutError as e:
    print(f"Operation timed out: {e}")

except DPBudgetExceededError as e:
    print(f"Privacy budget exhausted: {e}")
    # Save checkpoint before stopping
    tc.save_state()

except RateLimitedError as e:
    print(f"Rate limited, retry after: {e.retry_after}s")

except TGTinkerError as e:
    print(f"API error [{e.code}]: {e.message}")
```

## Next Steps

- [Training Guide](guides/training.md) - Advanced training workflows
- [Privacy Guide](guides/privacy.md) - DP and homomorphic encryption
- [API Reference](api-reference/service-client.md) - Full API documentation

# Training Guide

This guide covers training workflows and best practices with TenSafe.

## Training Architecture

TenSafe uses an asynchronous training model where operations are queued and executed on remote compute:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   Queue     │────▶│   Compute   │
│   SDK       │◀────│   Service   │◀────│   Backend   │
└─────────────┘     └─────────────┘     └─────────────┘
      │                                        │
      │            FutureHandle                │
      └────────────────────────────────────────┘
```

## Core Training Primitives

### forward_backward()

Computes the forward pass and backpropagates gradients:

```python
from tg_tinker import BatchData

# Using BatchData object
batch = BatchData(
    input_ids=[[1, 2, 3, 4], [5, 6, 7, 8]],
    attention_mask=[[1, 1, 1, 1], [1, 1, 1, 1]],
    labels=[[2, 3, 4, -100], [6, 7, 8, -100]],
)

future = tc.forward_backward(batch)

# Or using dict
future = tc.forward_backward({
    "input_ids": tokens,
    "attention_mask": mask,
    "labels": labels,
})

# Get result
result = future.result()
print(f"Loss: {result.loss}")
print(f"Grad norm: {result.grad_norm}")
print(f"Tokens processed: {result.tokens_processed}")
```

### optim_step()

Applies accumulated gradients with the optimizer:

```python
# Basic optimizer step
opt_future = tc.optim_step()
opt_result = opt_future.result()

print(f"New step: {opt_result.step}")
print(f"Learning rate: {opt_result.learning_rate}")

# With DP noise (when DP is enabled)
opt_future = tc.optim_step(apply_dp_noise=True)
opt_result = opt_future.result()

if opt_result.dp_metrics:
    print(f"Epsilon spent: {opt_result.dp_metrics.epsilon_spent}")
```

## Async Execution Patterns

### Overlapping Operations

Queue multiple operations before waiting:

```python
# Queue forward-backward immediately
fb_future = tc.forward_backward(batch)

# Queue optim step without waiting
opt_future = tc.optim_step()

# Now wait for results
fb_result = fb_future.result()
opt_result = opt_future.result()
```

### Pipelining Batches

Process multiple batches in a pipeline:

```python
futures = []

for i, batch in enumerate(batches):
    # Submit forward-backward
    fb_future = tc.forward_backward(batch)
    futures.append(("fb", i, fb_future))

    # Submit optim step
    opt_future = tc.optim_step()
    futures.append(("opt", i, opt_future))

# Collect all results
for op_type, batch_idx, future in futures:
    result = future.result()
    if op_type == "fb":
        print(f"Batch {batch_idx}: loss={result.loss:.4f}")
```

### Timeouts and Cancellation

```python
from tg_tinker import FutureTimeoutError

future = tc.forward_backward(batch)

try:
    # Wait with timeout
    result = future.result(timeout=120)
except FutureTimeoutError:
    # Cancel if taking too long
    cancelled = future.cancel()
    print(f"Operation cancelled: {cancelled}")
```

## LoRA Configuration

### Basic LoRA Setup

```python
from tg_tinker import LoRAConfig

lora = LoRAConfig(
    rank=16,           # Rank of low-rank matrices
    alpha=32,          # Scaling factor (typically 2x rank)
    dropout=0.05,      # Dropout rate
    target_modules=[   # Modules to apply LoRA
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ],
    bias="none",       # "none", "all", or "lora_only"
)
```

### Choosing LoRA Rank

| Rank | Memory | Quality | Use Case |
|------|--------|---------|----------|
| 4-8 | Low | Good | Simple tasks, quick experiments |
| 16-32 | Medium | Very Good | General fine-tuning |
| 64-128 | High | Excellent | Complex tasks, domain adaptation |
| 256+ | Very High | Marginal gains | Specialized research |

### Full Fine-tuning

For full parameter updates, omit lora_config:

```python
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=None,  # Full fine-tuning
    optimizer=OptimizerConfig(learning_rate=1e-5),
)
```

## Optimizer Configuration

### AdamW (Default)

```python
from tg_tinker import OptimizerConfig

optimizer = OptimizerConfig(
    name="adamw",
    learning_rate=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
)
```

### SGD with Momentum

```python
optimizer = OptimizerConfig(
    name="sgd",
    learning_rate=1e-3,
    weight_decay=0.0,
)
```

### Supported Optimizers

- `adamw` - AdamW with decoupled weight decay
- `adam` - Standard Adam
- `sgd` - Stochastic Gradient Descent
- `adafactor` - Memory-efficient Adam alternative

## Gradient Accumulation

Simulate larger batch sizes:

```python
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16),
    batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch size: 32
)

tc = service.create_training_client(config)

# Gradients accumulate across forward_backward calls
for micro_batch in micro_batches:
    tc.forward_backward(micro_batch).result()

# Single optimizer step uses accumulated gradients
tc.optim_step().result()
```

## Checkpoint Management

### Saving Checkpoints

```python
# Save full state
checkpoint = tc.save_state(
    include_optimizer=True,
    metadata={
        "epoch": 1,
        "validation_loss": 0.42,
        "dataset_version": "v2",
    }
)

print(f"Artifact ID: {checkpoint.artifact_id}")
print(f"Size: {checkpoint.size_bytes} bytes")
print(f"Hash: {checkpoint.content_hash}")
```

### Loading Checkpoints

```python
# Resume from checkpoint
result = tc.load_state(artifact_id="art-xxx-yyy")

print(f"Restored to step: {result.step}")
print(f"Status: {result.status}")
```

### Downloading Artifacts

```python
# Download encrypted artifact
data = service.pull_artifact(checkpoint.artifact_id)

# Save locally (still encrypted)
with open("checkpoint.bin", "wb") as f:
    f.write(data)
```

## Training Client State

### Monitoring Progress

```python
# Check current state
print(f"Step: {tc.step}")
print(f"Status: {tc.status}")

# Refresh from server
tc.refresh()
print(f"Updated step: {tc.step}")
```

### Client Status Values

| Status | Description |
|--------|-------------|
| `INITIALIZING` | Client being set up |
| `READY` | Ready for operations |
| `BUSY` | Processing an operation |
| `ERROR` | Error state |
| `TERMINATED` | Client shut down |

## Best Practices

### 1. Use Appropriate Batch Sizes

```python
# Start conservative, increase gradually
config = TrainingConfig(
    batch_size=4,
    gradient_accumulation_steps=8,  # Effective: 32
)
```

### 2. Monitor Loss and Gradients

```python
for batch in dataloader:
    result = tc.forward_backward(batch).result()

    # Check for training issues
    if result.loss > 10.0:
        print("Warning: High loss detected")
    if result.grad_norm > 100.0:
        print("Warning: Gradient explosion")
```

### 3. Regular Checkpointing

```python
checkpoint_every = 100

for step, batch in enumerate(dataloader):
    tc.forward_backward(batch).result()
    tc.optim_step().result()

    if (step + 1) % checkpoint_every == 0:
        checkpoint = tc.save_state()
        print(f"Checkpoint saved: {checkpoint.artifact_id}")
```

### 4. Handle Errors Gracefully

```python
from tg_tinker import TGTinkerError

try:
    for batch in dataloader:
        tc.forward_backward(batch).result()
        tc.optim_step().result()
except TGTinkerError as e:
    print(f"Training error: {e}")
    # Save emergency checkpoint
    tc.save_state(metadata={"error": str(e)})
    raise
```

## Next Steps

- [Sampling Guide](sampling.md) - Text generation
- [Privacy Guide](privacy.md) - DP and homomorphic encryption
- [API Reference](../api-reference/training-client.md) - Full TrainingClient API

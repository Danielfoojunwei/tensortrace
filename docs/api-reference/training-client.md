# TrainingClient

Client for managing training operations.

## Overview

`TrainingClient` provides methods for training loops, sampling, and checkpoint management. Created via `ServiceClient.create_training_client()`.

```python
from tg_tinker import ServiceClient, TrainingConfig, LoRAConfig

service = ServiceClient()
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16),
)

tc = service.create_training_client(config)
```

## Properties

### id

```python
@property
def id(self) -> str
```

Unique identifier for this training client.

### step

```python
@property
def step(self) -> int
```

Current training step (number of optimizer steps).

### config

```python
@property
def config(self) -> TrainingConfig
```

The training configuration.

### status

```python
@property
def status(self) -> str
```

Current status: "active", "completed", or "failed".

## Training Methods

### forward_backward

Execute forward and backward pass.

```python
def forward_backward(
    self,
    batch: Union[BatchData, Dict[str, Any]],
    batch_hash: Optional[str] = None,
) -> FutureHandle
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `batch` | `BatchData` or `dict` | Training batch with input_ids, attention_mask, labels |
| `batch_hash` | `str` | Optional hash for reproducibility tracking |

#### Returns

`FutureHandle` - Resolves to `ForwardBackwardResult`.

#### Example

```python
batch = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels": labels,
}

future = tc.forward_backward(batch)
result = future.result()

print(f"Loss: {result.loss}")
print(f"Gradient hash: {result.gradient_hash}")
```

### optim_step

Execute optimizer step.

```python
def optim_step(
    self,
    apply_dp_noise: bool = True,
) -> FutureHandle
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `apply_dp_noise` | `bool` | `True` | Apply DP noise if DP is enabled |

#### Returns

`FutureHandle` - Resolves to `OptimStepResult`.

#### Example

```python
future = tc.optim_step(apply_dp_noise=True)
result = future.result()

print(f"Gradient norm: {result.grad_norm}")
print(f"Learning rate: {result.learning_rate}")

if result.dp_metrics:
    print(f"Epsilon spent: {result.dp_metrics.epsilon_spent}")
```

### train

Set training mode.

```python
def train(self) -> None
```

#### Example

```python
tc.train()
for batch in train_loader:
    tc.forward_backward(batch).result()
    tc.optim_step().result()
```

### eval

Set evaluation mode.

```python
def eval(self) -> None
```

#### Example

```python
tc.eval()
for batch in val_loader:
    result = tc.forward_backward(batch).result()
    val_losses.append(result.loss)
```

## Sampling Methods

### sample

Generate text completions.

```python
def sample(
    self,
    prompts: Union[str, List[str]],
    max_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
    num_completions: int = 1,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
) -> SampleResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompts` | `str` or `List[str]` | required | Input prompt(s) |
| `max_tokens` | `int` | `128` | Maximum tokens to generate |
| `temperature` | `float` | `1.0` | Sampling temperature |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold |
| `top_k` | `int` | `None` | Top-k sampling limit |
| `stop_sequences` | `List[str]` | `None` | Stop generation sequences |
| `num_completions` | `int` | `1` | Completions per prompt |
| `repetition_penalty` | `float` | `1.0` | Penalty for repetition |
| `presence_penalty` | `float` | `0.0` | Penalty for token presence |
| `frequency_penalty` | `float` | `0.0` | Penalty for token frequency |

#### Returns

`SampleResult` - Contains list of samples.

#### Example

```python
result = tc.sample(
    prompts=["Once upon a time", "In a galaxy far away"],
    max_tokens=100,
    temperature=0.7,
    top_p=0.9,
)

for sample in result.samples:
    print(f"Prompt: {sample.prompt}")
    print(f"Completion: {sample.completion}")
    print(f"Tokens: {sample.num_tokens}")
```

### sample_stream

Stream text generation.

```python
def sample_stream(
    self,
    prompts: str,
    max_tokens: int = 128,
    temperature: float = 1.0,
    **kwargs,
) -> Iterator[StreamChunk]
```

#### Parameters

Same as `sample()`, but `prompts` must be a single string.

#### Yields

`StreamChunk` - Chunks of generated text.

#### Example

```python
for chunk in tc.sample_stream(
    prompts="Tell me a story",
    max_tokens=500,
):
    print(chunk.text, end="", flush=True)
```

## Checkpoint Methods

### save_state

Save training state.

```python
def save_state(
    self,
    include_optimizer: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> SaveStateResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_optimizer` | `bool` | `True` | Include optimizer state |
| `metadata` | `dict` | `None` | Custom metadata to store |

#### Returns

`SaveStateResult` - Checkpoint information.

#### Example

```python
checkpoint = tc.save_state(
    include_optimizer=True,
    metadata={
        "epoch": 5,
        "val_loss": 0.42,
        "dataset": "my-dataset-v2",
    }
)

print(f"Artifact ID: {checkpoint.artifact_id}")
print(f"Size: {checkpoint.size_bytes}")
print(f"Encryption: {checkpoint.encryption.algorithm}")
```

### load_state

Load training state.

```python
def load_state(
    self,
    artifact_id: str,
) -> LoadStateResult
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `artifact_id` | `str` | Checkpoint artifact ID |

#### Returns

`LoadStateResult` - Load operation result.

#### Example

```python
result = tc.load_state(checkpoint.artifact_id)

print(f"Loaded step: {tc.step}")
print(f"Loaded metadata: {result.metadata}")
```

## Result Types

### ForwardBackwardResult

```python
@dataclass
class ForwardBackwardResult:
    loss: float                      # Training loss
    batch_hash: str                  # Hash of input batch
    gradient_hash: str               # Hash of computed gradients
    num_tokens: int                  # Tokens processed
    latency_ms: float                # Operation latency
```

### OptimStepResult

```python
@dataclass
class OptimStepResult:
    grad_norm: float                 # Gradient norm before clipping
    learning_rate: float             # Current learning rate
    step: int                        # Current step number
    dp_metrics: Optional[DPMetrics]  # DP metrics if enabled
    latency_ms: float                # Operation latency
```

### DPMetrics

```python
@dataclass
class DPMetrics:
    noise_applied: bool              # Whether noise was applied
    epsilon_spent: float             # Epsilon this step
    total_epsilon: float             # Total epsilon spent
    delta: float                     # Current delta
    grad_norm_before_clip: float     # Norm before clipping
    grad_norm_after_clip: float      # Norm after clipping
    num_clipped: int                 # Samples clipped
```

### SampleResult

```python
@dataclass
class SampleResult:
    samples: List[Sample]            # Generated samples
    total_tokens: int                # Total tokens generated
    latency_ms: float                # Total latency
```

### Sample

```python
@dataclass
class Sample:
    prompt: str                      # Input prompt
    completion: str                  # Generated text
    num_tokens: int                  # Tokens in completion
    finish_reason: str               # "length", "stop", or "eos"
    latency_ms: float                # Generation latency
```

### SaveStateResult

```python
@dataclass
class SaveStateResult:
    artifact_id: str                 # Unique artifact ID
    size_bytes: int                  # Checkpoint size
    encryption: EncryptionInfo       # Encryption details
    metadata: Dict[str, Any]         # User metadata
    timestamp: datetime              # Save timestamp
```

### LoadStateResult

```python
@dataclass
class LoadStateResult:
    artifact_id: str                 # Loaded artifact ID
    step: int                        # Restored step
    metadata: Dict[str, Any]         # Stored metadata
    timestamp: datetime              # Original save time
```

## Error Handling

```python
from tg_tinker import (
    TGTinkerError,
    DPBudgetExceededError,
    FutureTimeoutError,
    InvalidRequestError,
)

try:
    result = tc.forward_backward(batch).result(timeout=60)
except FutureTimeoutError:
    print("Operation timed out")
except DPBudgetExceededError as e:
    print(f"DP budget exceeded: {e}")
    tc.save_state()  # Save before stopping
except InvalidRequestError as e:
    print(f"Invalid batch: {e}")
except TGTinkerError as e:
    print(f"API error: {e}")
```

## See Also

- [ServiceClient](service-client.md) - Main entry point
- [FutureHandle](futures.md) - Async operations
- [Configuration](configuration.md) - Training config options

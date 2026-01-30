# Configuration

Configuration options for TenSafe SDK.

## TenSafeConfig

Global SDK configuration.

```python
from tg_tinker import TenSafeConfig

config = TenSafeConfig(
    api_key="ts-your-key",
    base_url="https://api.tensafe.dev",
    timeout=300.0,
)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api_key` | `str` | `None` | API key for authentication |
| `base_url` | `str` | `https://api.tensafe.dev` | API base URL |
| `tenant_id` | `str` | `None` | Tenant ID |
| `timeout` | `float` | `300.0` | Request timeout (seconds) |
| `retry_count` | `int` | `3` | Max retries |
| `retry_backoff` | `float` | `1.0` | Retry backoff base (seconds) |
| `verify_ssl` | `bool` | `True` | Verify SSL certificates |
| `poll_interval` | `float` | `1.0` | Future polling interval |

### Environment Variables

| Variable | Field | Description |
|----------|-------|-------------|
| `TS_API_KEY` | `api_key` | API key |
| `TS_BASE_URL` | `base_url` | Base URL |
| `TS_TENANT_ID` | `tenant_id` | Tenant ID |
| `TS_TIMEOUT` | `timeout` | Timeout |
| `TS_RETRY_COUNT` | `retry_count` | Max retries |
| `TS_VERIFY_SSL` | `verify_ssl` | SSL verification |

## TrainingConfig

Configuration for training jobs.

```python
from tg_tinker import TrainingConfig, LoRAConfig, OptimizerConfig, DPConfig

config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16),
    optimizer=OptimizerConfig(learning_rate=1e-4),
    dp_config=DPConfig(target_epsilon=8.0),
    batch_size=8,
)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_ref` | `str` | required | Model identifier |
| `lora_config` | `LoRAConfig` | required | LoRA configuration |
| `optimizer` | `OptimizerConfig` | `None` | Optimizer settings |
| `dp_config` | `DPConfig` | `None` | Differential privacy settings |
| `batch_size` | `int` | `8` | Training batch size |
| `gradient_accumulation_steps` | `int` | `1` | Gradient accumulation |
| `max_sequence_length` | `int` | `2048` | Max sequence length |
| `mixed_precision` | `str` | `"bf16"` | "fp32", "fp16", "bf16" |
| `gradient_checkpointing` | `bool` | `False` | Enable gradient checkpointing |

## LoRAConfig

Low-Rank Adaptation configuration.

```python
from tg_tinker import LoRAConfig

lora_config = LoRAConfig(
    rank=16,
    alpha=32,
    dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rank` | `int` | `16` | LoRA rank |
| `alpha` | `float` | `32` | Scaling factor |
| `dropout` | `float` | `0.0` | Dropout probability |
| `target_modules` | `List[str]` | `["q_proj", "v_proj"]` | Modules to adapt |
| `bias` | `str` | `"none"` | "none", "all", "lora_only" |
| `modules_to_save` | `List[str]` | `[]` | Additional modules to save |

### Target Modules

Common target module configurations:

```python
# Attention only (default)
target_modules=["q_proj", "v_proj"]

# Full attention
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

# Attention + MLP
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
```

## OptimizerConfig

Optimizer configuration.

```python
from tg_tinker import OptimizerConfig

optimizer = OptimizerConfig(
    name="adamw",
    learning_rate=1e-4,
    weight_decay=0.01,
)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `"adamw"` | Optimizer name |
| `learning_rate` | `float` | `1e-4` | Learning rate |
| `weight_decay` | `float` | `0.0` | Weight decay |
| `betas` | `Tuple[float, float]` | `(0.9, 0.999)` | Adam betas |
| `eps` | `float` | `1e-8` | Numerical stability |
| `warmup_steps` | `int` | `0` | LR warmup steps |
| `lr_scheduler` | `str` | `"constant"` | Scheduler type |
| `lr_scheduler_kwargs` | `dict` | `{}` | Scheduler arguments |

### Supported Optimizers

| Name | Description |
|------|-------------|
| `adamw` | AdamW (default) |
| `adam` | Adam |
| `sgd` | SGD with momentum |
| `adafactor` | Adafactor (memory efficient) |

### LR Schedulers

| Name | Description |
|------|-------------|
| `constant` | Constant LR (default) |
| `linear` | Linear decay |
| `cosine` | Cosine annealing |
| `cosine_with_restarts` | Cosine with warm restarts |

## DPConfig

Differential privacy configuration.

```python
from tg_tinker import DPConfig

dp_config = DPConfig(
    enabled=True,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    target_epsilon=8.0,
    target_delta=1e-5,
)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable DP |
| `noise_multiplier` | `float` | `1.0` | Noise scale |
| `max_grad_norm` | `float` | `1.0` | Gradient clipping |
| `target_epsilon` | `float` | `8.0` | Privacy budget |
| `target_delta` | `float` | `1e-5` | Failure probability |
| `accountant_type` | `str` | `"rdp"` | Accounting method |
| `secure_mode` | `bool` | `False` | Cryptographic noise |

### Accountant Types

| Type | Description |
|------|-------------|
| `rdp` | Renyi DP (default, tightest) |
| `gdp` | Gaussian DP |
| `prv` | Privacy Random Variable |

### Privacy Presets

```python
# Strong privacy (medical/financial)
strong = DPConfig(
    enabled=True,
    noise_multiplier=1.5,
    max_grad_norm=0.5,
    target_epsilon=3.0,
)

# Moderate privacy (production)
moderate = DPConfig(
    enabled=True,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    target_epsilon=8.0,
)

# Relaxed privacy (research)
relaxed = DPConfig(
    enabled=True,
    noise_multiplier=0.5,
    max_grad_norm=1.5,
    target_epsilon=15.0,
)
```

## HESchemeParams

Homomorphic encryption parameters.

```python
from tensorguard.n2he import HESchemeParams, N2HEScheme

params = HESchemeParams(
    scheme=N2HEScheme.CKKS,
    poly_modulus_degree=8192,
    security_level=128,
)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `scheme` | `N2HEScheme` | `CKKS` | HE scheme |
| `poly_modulus_degree` | `int` | `8192` | Polynomial degree |
| `security_level` | `int` | `128` | Security bits |
| `scale` | `float` | `2^40` | CKKS scale |

### Schemes

| Scheme | Description |
|--------|-------------|
| `LWE` | Learning With Errors |
| `RLWE` | Ring-LWE |
| `CKKS` | Approximate arithmetic (default) |

## AdapterEncryptionConfig

Configuration for encrypted LoRA adapters.

```python
from tensorguard.n2he import AdapterEncryptionConfig

config = AdapterEncryptionConfig(
    rank=16,
    encrypted_layers=["q_proj", "v_proj"],
    batch_encryption=True,
)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rank` | `int` | `16` | LoRA rank |
| `encrypted_layers` | `List[str]` | `[]` | Layers to encrypt |
| `batch_encryption` | `bool` | `True` | Batch encrypt operations |
| `relinearization_freq` | `int` | `10` | Relinearization frequency |

## Configuration Validation

Configurations are validated on creation:

```python
from tg_tinker import TrainingConfig, LoRAConfig
from pydantic import ValidationError

try:
    config = TrainingConfig(
        model_ref="meta-llama/Llama-3-8B",
        lora_config=LoRAConfig(rank=-1),  # Invalid!
    )
except ValidationError as e:
    print(f"Invalid config: {e}")
```

## Loading from File

```python
import yaml
from tg_tinker import TrainingConfig

# Load from YAML
with open("config.yaml") as f:
    config_dict = yaml.safe_load(f)

config = TrainingConfig(**config_dict)
```

Example `config.yaml`:

```yaml
model_ref: meta-llama/Llama-3-8B
batch_size: 8
gradient_accumulation_steps: 4

lora_config:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj

optimizer:
  name: adamw
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_steps: 100
  lr_scheduler: cosine

dp_config:
  enabled: true
  target_epsilon: 8.0
  noise_multiplier: 1.0
  max_grad_norm: 1.0
```

## See Also

- [ServiceClient](service-client.md) - Using configuration
- [TrainingClient](training-client.md) - Training with config
- [Privacy Guide](../guides/privacy.md) - DP configuration

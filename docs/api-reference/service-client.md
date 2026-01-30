# ServiceClient

The main entry point for the TenSafe SDK.

## Overview

`ServiceClient` provides access to the TenSafe API for creating training clients, managing artifacts, and accessing audit logs.

```python
from tg_tinker import ServiceClient

service = ServiceClient(api_key="ts-your-key")
```

## Constructor

```python
ServiceClient(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    tenant_id: Optional[str] = None,
    timeout: float = 300.0,
    retry_count: int = 3,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `None` | API key. Falls back to `TS_API_KEY` env var |
| `base_url` | `str` | `https://api.tensafe.dev` | API base URL |
| `tenant_id` | `str` | `None` | Tenant ID (derived from API key if not set) |
| `timeout` | `float` | `300.0` | Request timeout in seconds |
| `retry_count` | `int` | `3` | Max retries for failed requests |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `TS_API_KEY` | Default API key |
| `TS_BASE_URL` | Default base URL |
| `TS_TENANT_ID` | Default tenant ID |
| `TS_TIMEOUT` | Default timeout |

## Methods

### create_training_client

Create a new training client for fine-tuning.

```python
def create_training_client(
    self,
    config: TrainingConfig,
    checkpoint_id: Optional[str] = None,
) -> TrainingClient
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `TrainingConfig` | Training configuration |
| `checkpoint_id` | `str` | Optional checkpoint to restore from |

#### Returns

`TrainingClient` - A new training client instance.

#### Example

```python
from tg_tinker import TrainingConfig, LoRAConfig

config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16),
)

tc = service.create_training_client(config)
print(f"Created client: {tc.id}")
```

### get_future

Retrieve a future by ID.

```python
def get_future(self, future_id: str) -> FutureHandle
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `future_id` | `str` | The future ID to retrieve |

#### Returns

`FutureHandle` - Handle for the async operation.

#### Example

```python
# Save future ID for later
future = tc.forward_backward(batch)
future_id = future.id

# Later: retrieve and check status
future = service.get_future(future_id)
if future.is_done():
    result = future.result()
```

### pull_artifact

Download an artifact.

```python
def pull_artifact(
    self,
    artifact_id: str,
    destination: Optional[str] = None,
) -> bytes
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `artifact_id` | `str` | Artifact ID to download |
| `destination` | `str` | Optional file path to save to |

#### Returns

`bytes` - The artifact data (encrypted).

#### Example

```python
# Download to memory
data = service.pull_artifact(checkpoint.artifact_id)

# Download to file
service.pull_artifact(
    checkpoint.artifact_id,
    destination="./checkpoint.bin"
)
```

### get_audit_logs

Retrieve audit logs.

```python
def get_audit_logs(
    self,
    training_client_id: Optional[str] = None,
    operation: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100,
) -> List[AuditLogEntry]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `training_client_id` | `str` | Filter by training client |
| `operation` | `str` | Filter by operation type |
| `start_time` | `datetime` | Filter by start time |
| `end_time` | `datetime` | Filter by end time |
| `limit` | `int` | Maximum entries to return |

#### Returns

`List[AuditLogEntry]` - List of audit log entries.

#### Example

```python
from datetime import datetime, timedelta

# Get recent logs for a training client
logs = service.get_audit_logs(
    training_client_id=tc.id,
    start_time=datetime.now() - timedelta(hours=1),
    limit=50,
)

for entry in logs:
    print(f"{entry.timestamp}: {entry.operation} - {entry.success}")
```

### list_training_clients

List all training clients.

```python
def list_training_clients(
    self,
    status: Optional[str] = None,
    limit: int = 100,
) -> List[TrainingClientInfo]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | `str` | Filter by status ("active", "completed", "failed") |
| `limit` | `int` | Maximum entries to return |

#### Returns

`List[TrainingClientInfo]` - List of training client info.

#### Example

```python
# List active training clients
clients = service.list_training_clients(status="active")

for client in clients:
    print(f"{client.id}: {client.model_ref} - step {client.step}")
```

### delete_training_client

Delete a training client.

```python
def delete_training_client(self, training_client_id: str) -> None
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `training_client_id` | `str` | Training client ID to delete |

#### Example

```python
service.delete_training_client(tc.id)
```

## Properties

### tenant_id

```python
@property
def tenant_id(self) -> str
```

The tenant ID for this client.

### config

```python
@property
def config(self) -> TenSafeConfig
```

The configuration object.

## Error Handling

```python
from tg_tinker import (
    TGTinkerError,
    AuthenticationError,
    RateLimitedError,
)

try:
    service = ServiceClient(api_key="invalid")
    tc = service.create_training_client(config)
except AuthenticationError as e:
    print(f"Invalid API key: {e}")
except RateLimitedError as e:
    print(f"Rate limited: retry after {e.retry_after}s")
except TGTinkerError as e:
    print(f"API error: {e.code} - {e.message}")
```

## Thread Safety

`ServiceClient` is thread-safe. You can share a single instance across threads:

```python
import threading

service = ServiceClient()

def worker(batch):
    tc = service.create_training_client(config)
    tc.forward_backward(batch).result()

threads = [
    threading.Thread(target=worker, args=(batch,))
    for batch in batches
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

## See Also

- [TrainingClient](training-client.md) - Training operations
- [Configuration](configuration.md) - Configuration options
- [Exceptions](exceptions.md) - Error handling

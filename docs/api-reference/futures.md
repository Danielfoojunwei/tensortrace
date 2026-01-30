# FutureHandle

Handle for asynchronous operations.

## Overview

`FutureHandle` represents a pending async operation. Training operations like `forward_backward()` and `optim_step()` return futures that can be awaited or polled.

```python
# Queue operation
future = tc.forward_backward(batch)

# Wait for result
result = future.result()
```

## Properties

### id

```python
@property
def id(self) -> str
```

Unique identifier for this future.

### status

```python
@property
def status(self) -> str
```

Current status: "pending", "running", "completed", or "failed".

### operation

```python
@property
def operation(self) -> str
```

The operation type: "forward_backward", "optim_step", etc.

## Methods

### result

Wait for and return the result.

```python
def result(
    self,
    timeout: Optional[float] = None,
) -> Any
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | `float` | `None` | Max seconds to wait (None = use default) |

#### Returns

The operation result (type depends on operation).

#### Raises

- `FutureTimeoutError` - If timeout exceeded
- `FutureCancelledError` - If operation was cancelled
- `TGTinkerError` - If operation failed

#### Example

```python
future = tc.forward_backward(batch)

try:
    result = future.result(timeout=30)
    print(f"Loss: {result.loss}")
except FutureTimeoutError:
    print("Operation timed out")
```

### is_done

Check if operation is complete.

```python
def is_done(self) -> bool
```

#### Returns

`bool` - True if completed (success or failure).

#### Example

```python
future = tc.forward_backward(batch)

while not future.is_done():
    print("Still processing...")
    time.sleep(1)

result = future.result()
```

### is_success

Check if operation succeeded.

```python
def is_success(self) -> bool
```

#### Returns

`bool` - True if completed successfully.

### is_failed

Check if operation failed.

```python
def is_failed(self) -> bool
```

#### Returns

`bool` - True if operation failed.

### cancel

Cancel the operation.

```python
def cancel(self) -> bool
```

#### Returns

`bool` - True if cancellation was successful.

#### Example

```python
future = tc.forward_backward(batch)

# Cancel if taking too long
time.sleep(5)
if not future.is_done():
    cancelled = future.cancel()
    if cancelled:
        print("Operation cancelled")
```

### wait

Wait for completion without returning result.

```python
def wait(
    self,
    timeout: Optional[float] = None,
) -> bool
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | `float` | `None` | Max seconds to wait |

#### Returns

`bool` - True if completed within timeout.

#### Example

```python
future = tc.forward_backward(batch)

if future.wait(timeout=10):
    result = future.result()
else:
    print("Still running after 10s")
```

### get_error

Get error details if failed.

```python
def get_error(self) -> Optional[TGTinkerError]
```

#### Returns

`TGTinkerError` or `None` - Error if operation failed.

#### Example

```python
future = tc.forward_backward(bad_batch)
future.wait()

if future.is_failed():
    error = future.get_error()
    print(f"Failed: {error.code} - {error.message}")
```

## Patterns

### Fire and Forget

Queue multiple operations:

```python
futures = []
for batch in batches:
    futures.append(tc.forward_backward(batch))
    futures.append(tc.optim_step())

# Wait for all
for future in futures:
    future.result()
```

### Parallel Operations

```python
# Queue operations in parallel
fb_futures = [tc.forward_backward(b) for b in batches]

# Wait for all forward-backward
for future in fb_futures:
    future.result()

# Then optimizer step
tc.optim_step().result()
```

### Timeout with Retry

```python
max_retries = 3

for attempt in range(max_retries):
    future = tc.forward_backward(batch)
    try:
        result = future.result(timeout=30)
        break
    except FutureTimeoutError:
        future.cancel()
        if attempt == max_retries - 1:
            raise
        time.sleep(2 ** attempt)
```

### Progress Monitoring

```python
futures = []
total = len(dataloader)

for i, batch in enumerate(dataloader):
    futures.append(tc.forward_backward(batch))
    futures.append(tc.optim_step())

    # Check progress
    completed = sum(1 for f in futures if f.is_done())
    print(f"Progress: {completed}/{len(futures)}")
```

### Callback Pattern

```python
def on_complete(future):
    if future.is_success():
        result = future.result()
        print(f"Completed: loss={result.loss}")
    else:
        error = future.get_error()
        print(f"Failed: {error}")

# Poll in background
import threading

def poll_future(future, callback):
    while not future.is_done():
        time.sleep(0.1)
    callback(future)

future = tc.forward_backward(batch)
thread = threading.Thread(
    target=poll_future,
    args=(future, on_complete)
)
thread.start()
```

## Error Handling

```python
from tg_tinker import (
    FutureTimeoutError,
    FutureCancelledError,
    TGTinkerError,
)

future = tc.forward_backward(batch)

try:
    result = future.result(timeout=60)
except FutureTimeoutError:
    print("Operation timed out")
    future.cancel()
except FutureCancelledError:
    print("Operation was cancelled")
except TGTinkerError as e:
    print(f"Operation failed: {e.code} - {e.message}")
```

## Retrieving Futures

Futures can be retrieved later by ID:

```python
# Save future ID
future = tc.forward_backward(batch)
future_id = future.id

# Later: retrieve and check
future = service.get_future(future_id)
if future.is_done():
    result = future.result()
```

## See Also

- [TrainingClient](training-client.md) - Training operations
- [Exceptions](exceptions.md) - Error types

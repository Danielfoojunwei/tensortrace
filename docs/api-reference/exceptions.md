# Exceptions

Error types in the TenSafe SDK.

## Exception Hierarchy

```
TGTinkerError (base)
├── AuthenticationError
├── AuthorizationError
├── RateLimitedError
├── InvalidRequestError
├── ResourceNotFoundError
├── DPBudgetExceededError
├── FutureTimeoutError
├── FutureCancelledError
└── ServerError
```

## TGTinkerError

Base exception for all SDK errors.

```python
class TGTinkerError(Exception):
    code: str           # Error code
    message: str        # Error message
    details: dict       # Additional details
    request_id: str     # Request ID for debugging
```

### Example

```python
from tg_tinker import TGTinkerError

try:
    tc.forward_backward(batch).result()
except TGTinkerError as e:
    print(f"Error code: {e.code}")
    print(f"Message: {e.message}")
    print(f"Request ID: {e.request_id}")
    print(f"Details: {e.details}")
```

## AuthenticationError

Invalid or missing API key.

```python
class AuthenticationError(TGTinkerError):
    pass
```

### Common Causes

- Missing API key
- Invalid API key format
- Expired API key

### Example

```python
from tg_tinker import ServiceClient, AuthenticationError

try:
    service = ServiceClient(api_key="invalid-key")
    tc = service.create_training_client(config)
except AuthenticationError as e:
    print("Invalid API key. Check TS_API_KEY environment variable.")
```

## AuthorizationError

Permission denied for operation.

```python
class AuthorizationError(TGTinkerError):
    pass
```

### Common Causes

- Accessing another tenant's resources
- Missing required permissions
- Quota exceeded

### Example

```python
from tg_tinker import AuthorizationError

try:
    service.pull_artifact("other-tenant-artifact-id")
except AuthorizationError as e:
    print(f"Access denied: {e.message}")
```

## RateLimitedError

API rate limit exceeded.

```python
class RateLimitedError(TGTinkerError):
    retry_after: float  # Seconds to wait before retry
```

### Example

```python
from tg_tinker import RateLimitedError
import time

try:
    tc.forward_backward(batch).result()
except RateLimitedError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
    time.sleep(e.retry_after)
    # Retry operation
```

## InvalidRequestError

Invalid request parameters.

```python
class InvalidRequestError(TGTinkerError):
    field: str          # Field that failed validation
    constraint: str     # Validation constraint
```

### Common Causes

- Invalid batch format
- Missing required fields
- Invalid parameter values

### Example

```python
from tg_tinker import InvalidRequestError

try:
    tc.sample(prompts="Hello", max_tokens=-1)  # Invalid!
except InvalidRequestError as e:
    print(f"Invalid field '{e.field}': {e.constraint}")
```

## ResourceNotFoundError

Requested resource not found.

```python
class ResourceNotFoundError(TGTinkerError):
    resource_type: str  # Type of resource
    resource_id: str    # ID that was not found
```

### Example

```python
from tg_tinker import ResourceNotFoundError

try:
    tc.load_state("nonexistent-artifact-id")
except ResourceNotFoundError as e:
    print(f"{e.resource_type} not found: {e.resource_id}")
```

## DPBudgetExceededError

Differential privacy budget exhausted.

```python
class DPBudgetExceededError(TGTinkerError):
    current_epsilon: float   # Current total epsilon
    max_epsilon: float       # Maximum allowed epsilon
    delta: float             # Current delta
```

### Example

```python
from tg_tinker import DPBudgetExceededError

try:
    for batch in dataloader:
        tc.forward_backward(batch).result()
        tc.optim_step(apply_dp_noise=True).result()
except DPBudgetExceededError as e:
    print(f"DP budget exceeded!")
    print(f"Current epsilon: {e.current_epsilon}")
    print(f"Max epsilon: {e.max_epsilon}")

    # Save checkpoint before stopping
    tc.save_state(metadata={"reason": "dp_budget_exceeded"})
```

### Prevention

```python
# Monitor budget during training
for batch in dataloader:
    result = tc.optim_step(apply_dp_noise=True).result()

    if result.dp_metrics:
        epsilon = result.dp_metrics.total_epsilon
        max_epsilon = config.dp_config.target_epsilon

        # Stop before exceeding
        if epsilon > 0.9 * max_epsilon:
            print("Approaching budget limit, stopping early")
            break
```

## FutureTimeoutError

Async operation timed out.

```python
class FutureTimeoutError(TGTinkerError):
    future_id: str      # ID of the future
    timeout: float      # Timeout that was exceeded
```

### Example

```python
from tg_tinker import FutureTimeoutError

future = tc.forward_backward(large_batch)

try:
    result = future.result(timeout=30)
except FutureTimeoutError as e:
    print(f"Operation {e.future_id} timed out after {e.timeout}s")
    future.cancel()
```

## FutureCancelledError

Async operation was cancelled.

```python
class FutureCancelledError(TGTinkerError):
    future_id: str      # ID of the cancelled future
```

### Example

```python
from tg_tinker import FutureCancelledError

future = tc.forward_backward(batch)
future.cancel()

try:
    result = future.result()
except FutureCancelledError as e:
    print(f"Operation {e.future_id} was cancelled")
```

## ServerError

Internal server error.

```python
class ServerError(TGTinkerError):
    pass
```

### Example

```python
from tg_tinker import ServerError
import time

max_retries = 3

for attempt in range(max_retries):
    try:
        result = tc.forward_backward(batch).result()
        break
    except ServerError as e:
        if attempt == max_retries - 1:
            raise
        print(f"Server error, retrying... ({e.request_id})")
        time.sleep(2 ** attempt)
```

## Error Handling Patterns

### Comprehensive Handler

```python
from tg_tinker import (
    TGTinkerError,
    AuthenticationError,
    AuthorizationError,
    RateLimitedError,
    InvalidRequestError,
    ResourceNotFoundError,
    DPBudgetExceededError,
    FutureTimeoutError,
    FutureCancelledError,
    ServerError,
)

def safe_train_step(tc, batch):
    """Execute training step with comprehensive error handling."""
    try:
        fb_result = tc.forward_backward(batch).result(timeout=60)
        opt_result = tc.optim_step().result(timeout=30)
        return fb_result, opt_result

    except AuthenticationError:
        print("Authentication failed. Check API key.")
        raise SystemExit(1)

    except AuthorizationError as e:
        print(f"Permission denied: {e.message}")
        raise SystemExit(1)

    except RateLimitedError as e:
        time.sleep(e.retry_after)
        return safe_train_step(tc, batch)  # Retry

    except InvalidRequestError as e:
        print(f"Invalid request: {e.field} - {e.constraint}")
        raise

    except DPBudgetExceededError:
        print("DP budget exceeded. Saving checkpoint...")
        tc.save_state()
        raise

    except FutureTimeoutError:
        print("Operation timed out. Retrying...")
        return safe_train_step(tc, batch)

    except FutureCancelledError:
        print("Operation cancelled. Skipping batch.")
        return None, None

    except ServerError as e:
        print(f"Server error ({e.request_id}). Retrying...")
        time.sleep(5)
        return safe_train_step(tc, batch)

    except TGTinkerError as e:
        print(f"Unknown error: {e.code} - {e.message}")
        raise
```

### Retry with Backoff

```python
def with_retry(func, max_retries=3, backoff_base=2):
    """Retry function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitedError as e:
            time.sleep(e.retry_after)
        except (ServerError, FutureTimeoutError):
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff_base ** attempt)
    return func()  # Final attempt

# Usage
result = with_retry(lambda: tc.forward_backward(batch).result())
```

### Logging Errors

```python
import logging

logger = logging.getLogger(__name__)

def logged_operation(func):
    """Decorator to log errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TGTinkerError as e:
            logger.error(
                f"API error: {e.code}",
                extra={
                    "request_id": e.request_id,
                    "message": e.message,
                    "details": e.details,
                }
            )
            raise
    return wrapper

@logged_operation
def train_step(tc, batch):
    return tc.forward_backward(batch).result()
```

## See Also

- [ServiceClient](service-client.md) - Client operations
- [FutureHandle](futures.md) - Async operations

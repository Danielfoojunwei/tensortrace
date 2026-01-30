# Installation

## Requirements

- Python 3.9+
- pip or uv package manager

## Install from PyPI

```bash
pip install tg-tinker
```

Or using uv:

```bash
uv add tg-tinker
```

## Install from Source

```bash
git clone https://github.com/your-org/tensafe.git
cd tensafe
pip install -e .
```

## Configuration

### Environment Variables

Set your API key and configuration via environment variables:

```bash
# Required
export TS_API_KEY="ts-your-api-key-here"

# Optional
export TS_BASE_URL="https://api.tensafe.dev"
export TS_TENANT_ID="your-tenant-id"
export TS_TIMEOUT="300"
export TS_RETRY_COUNT="3"
```

### Using a .env File

Create a `.env` file in your project root:

```bash
# .env
TS_API_KEY=ts-your-api-key-here
TS_BASE_URL=https://api.tensafe.dev
TS_TENANT_ID=your-tenant-id
```

The SDK automatically loads from `.env` files.

### Programmatic Configuration

Override environment variables in code:

```python
from tg_tinker import ServiceClient

client = ServiceClient(
    api_key="ts-your-api-key",
    base_url="https://api.tensafe.dev",
    timeout=600.0,
)
```

## Verify Installation

```python
from tg_tinker import ServiceClient, __version__

print(f"TenSafe SDK version: {__version__}")

# Test connection (requires valid API key)
client = ServiceClient()
print("Connected successfully!")
```

## Dependencies

The SDK has minimal dependencies:

- `httpx` - HTTP client
- `pydantic` - Data validation
- `pydantic-settings` - Configuration management

### Optional Dependencies

For N2HE homomorphic encryption:

```bash
pip install tg-tinker[n2he]
```

For development:

```bash
pip install tg-tinker[dev]
```

## Next Steps

- [Quickstart](quickstart.md) - Run your first training job
- [Configuration](api-reference/configuration.md) - Advanced configuration options

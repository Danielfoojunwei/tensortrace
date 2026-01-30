# Cookbook: Encrypted Inference with N2HE

Run private inference using homomorphic encryption.

## Overview

N2HE (Neural Network Homomorphic Encryption) enables computation on encrypted data. This tutorial shows how to:

- Generate HE keys
- Encrypt inputs for inference
- Run encrypted LoRA computations
- Decrypt outputs

## Prerequisites

```bash
pip install tg-tinker numpy
export TS_API_KEY=ts-your-key
```

## Key Concepts

### Homomorphic Encryption

HE allows arithmetic operations on ciphertexts:
- `Enc(a) + Enc(b) = Enc(a + b)`
- `Enc(a) * Enc(b) = Enc(a * b)`

TenSafe uses CKKS scheme for approximate arithmetic on real numbers.

### Security Levels

| Level | Description | Key Size |
|-------|-------------|----------|
| 128-bit | Standard (recommended) | ~8KB |
| 192-bit | High security | ~16KB |
| 256-bit | Post-quantum | ~32KB |

## Step 1: Initialize HE Context

```python
from tensorguard.n2he import (
    HEKeyManager,
    HESchemeParams,
    N2HEScheme,
    N2HEContext,
)

# Create key manager
key_manager = HEKeyManager(tenant_id="my-tenant")

# Configure HE parameters
params = HESchemeParams(
    scheme=N2HEScheme.CKKS,
    poly_modulus_degree=8192,
    security_level=128,
)

# Generate key bundle
bundle = key_manager.generate_key_bundle(params=params)

print(f"Public key: {len(bundle.public_key.data)} bytes")
print(f"Secret key: {len(bundle.secret_key.data)} bytes")
print(f"Eval key: {len(bundle.evaluation_key.data)} bytes")
print(f"Total: {bundle.get_total_size()} bytes")
```

## Step 2: Create Encryption Context

```python
# Create context for encryption/decryption
context = N2HEContext(params=params)

# Load keys
context.load_public_key(bundle.public_key)
context.load_secret_key(bundle.secret_key)
context.load_evaluation_key(bundle.evaluation_key)
```

## Step 3: Encrypt Data

```python
import numpy as np

# Sample embedding vector
embedding = np.random.randn(512).astype(np.float32)

# Encrypt
ciphertext = context.encrypt(embedding)

print(f"Original shape: {embedding.shape}")
print(f"Ciphertext size: {len(ciphertext.data)} bytes")
print(f"Expansion ratio: {len(ciphertext.data) / (embedding.nbytes):.1f}x")
```

## Step 4: HE Operations

### Addition

```python
# Encrypt two vectors
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

ct_a = context.encrypt(a)
ct_b = context.encrypt(b)

# Encrypted addition
ct_sum = context.add(ct_a, ct_b)

# Decrypt result
result = context.decrypt(ct_sum)
print(f"a + b = {result[:3]}")  # [5.0, 7.0, 9.0]
```

### Multiplication

```python
# Encrypted multiplication
ct_prod = context.multiply(ct_a, ct_b)

# Decrypt
result = context.decrypt(ct_prod)
print(f"a * b = {result[:3]}")  # [4.0, 10.0, 18.0]
```

### Scalar Operations

```python
scalar = np.array([2.0], dtype=np.float32)

# Multiply by scalar
ct_scaled = context.multiply(ct_a, context.encrypt(scalar))

result = context.decrypt(ct_scaled)
print(f"2 * a = {result[:3]}")  # [2.0, 4.0, 6.0]
```

### Matrix Multiplication

```python
# For matrix operations, need evaluation key
ek = context.export_eval_key()

# Matrix (simulated LoRA weights)
W = np.random.randn(3, 3).astype(np.float32)
x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# Encrypt input
ct_x = context.encrypt(x)

# Encrypted matrix multiply
ct_result = context.matmul(ct_x, W, ek)

# Decrypt
result = context.decrypt(ct_result)
expected = W @ x
print(f"Wx (encrypted): {result[:3]}")
print(f"Wx (expected): {expected}")
```

## Step 5: Encrypted LoRA Runtime

```python
from tensorguard.n2he import (
    EncryptedLoRARuntime,
    AdapterEncryptionConfig,
    create_encrypted_runtime,
)

# Configure encrypted LoRA
adapter_config = AdapterEncryptionConfig(
    rank=16,
    encrypted_layers=["q_proj", "v_proj"],
    batch_encryption=True,
)

# Create runtime
runtime = create_encrypted_runtime(
    config=adapter_config,
    key_bundle=bundle,
)

# Simulate LoRA forward pass
base_weights = np.random.randn(16, 768).astype(np.float32)

# Compute encrypted delta
encrypted_delta = runtime.forward(base_weights)

print(f"Encrypted LoRA delta computed")
print(f"Output shape: {len(encrypted_delta)}")
```

## Step 6: Private Inference Mode

```python
from tensorguard.n2he import (
    PrivateInferenceMode,
    create_private_inference_mode,
)

# Create private inference mode
inference = create_private_inference_mode(
    key_bundle=bundle,
    params=params,
)

# Encrypt input embedding
input_embedding = np.random.randn(512).astype(np.float32)
encrypted_input = inference.encrypt_input(input_embedding)

# Process (server computes on encrypted data)
encrypted_output = inference.process(encrypted_input)

# Decrypt output
output = inference.decrypt_output(encrypted_output)

print(f"Input shape: {input_embedding.shape}")
print(f"Output shape: {output.shape}")
```

## Step 7: Serialization

```python
from tensorguard.n2he import (
    CiphertextFormat,
    serialize_ciphertext,
    deserialize_ciphertext,
    create_ciphertext_bundle,
)

# Serialize ciphertext for storage/transmission
serialized = serialize_ciphertext(
    ciphertext,
    format=CiphertextFormat.BINARY,
)

print(f"Format: {serialized.format}")
print(f"Size: {len(serialized.data)} bytes")

# Deserialize
restored = deserialize_ciphertext(serialized, params)

# Verify
original = context.decrypt(ciphertext)
restored_data = context.decrypt(restored)
print(f"Match: {np.allclose(original, restored_data)}")
```

### Serialization Formats

| Format | Size | Speed | Use Case |
|--------|------|-------|----------|
| BINARY | Smallest | Fastest | Storage, transmission |
| JSON | Larger | Slow | Debugging, inspection |
| BASE64 | Medium | Medium | Text protocols |
| CBOR | Small | Fast | Structured data |

```python
# Compare formats
for fmt in [CiphertextFormat.BINARY, CiphertextFormat.JSON, CiphertextFormat.BASE64]:
    ser = serialize_ciphertext(ciphertext, format=fmt)
    print(f"{fmt.name}: {len(ser.data)} bytes")
```

## Step 8: Ciphertext Bundles

```python
# Bundle multiple ciphertexts
bundle = create_ciphertext_bundle(
    ciphertexts=[ct_a, ct_b, ct_sum],
    metadata={"operation": "addition_example"},
)

print(f"Bundle contains {len(bundle.ciphertexts)} ciphertexts")
print(f"Total size: {bundle.get_total_size()} bytes")

# Serialize entire bundle
serialized_bundle = bundle.serialize(format=CiphertextFormat.BINARY)
```

## Complete Example: Private Query

```python
#!/usr/bin/env python
"""Private query with encrypted embedding."""

import numpy as np
from tensorguard.n2he import (
    HEKeyManager,
    HESchemeParams,
    N2HEScheme,
    N2HEContext,
    create_private_inference_mode,
    serialize_ciphertext,
    CiphertextFormat,
)


def main():
    # Setup
    tenant_id = "my-tenant"
    key_manager = HEKeyManager(tenant_id=tenant_id)

    params = HESchemeParams(
        scheme=N2HEScheme.CKKS,
        poly_modulus_degree=8192,
        security_level=128,
    )

    bundle = key_manager.generate_key_bundle(params=params)

    # Create inference mode
    inference = create_private_inference_mode(
        key_bundle=bundle,
        params=params,
    )

    # Simulate query embedding (e.g., from a local embedding model)
    query = "What is the capital of France?"
    # In practice: query_embedding = embed(query)
    query_embedding = np.random.randn(512).astype(np.float32)

    # Encrypt query
    print("Encrypting query...")
    encrypted_query = inference.encrypt_input(query_embedding)

    # Serialize for transmission
    serialized = serialize_ciphertext(
        encrypted_query,
        format=CiphertextFormat.BINARY,
    )
    print(f"Encrypted query size: {len(serialized.data)} bytes")

    # --- Server side (would happen on TenSafe) ---
    # encrypted_response = model.forward(encrypted_query)

    # Simulate encrypted response
    encrypted_response = inference.process(encrypted_query)

    # --- Client side ---
    # Decrypt response
    print("Decrypting response...")
    response_embedding = inference.decrypt_output(encrypted_response)

    print(f"Response embedding shape: {response_embedding.shape}")
    print("Query processed with full privacy!")


if __name__ == "__main__":
    main()
```

## Performance Benchmarks

From E2E tests:

| Operation | Time (ms) | Throughput |
|-----------|-----------|------------|
| Key Generation | 0.114 | 8,751 ops/sec |
| Encryption | 0.022 | 46,014 ops/sec |
| Decryption | 0.019 | 52,000 ops/sec |
| LoRA Delta | 0.676 | 1,479 ops/sec |
| Add | 0.008 | 125,000 ops/sec |
| Multiply | 0.045 | 22,222 ops/sec |

## Best Practices

### 1. Key Management

```python
# Generate keys once, store securely
bundle = key_manager.generate_key_bundle(params=params)

# Store encrypted secret key
encrypted_sk = key_manager.export_encrypted_secret_key(
    bundle.secret_key,
    password="strong-password"
)

# Never transmit secret key unencrypted!
```

### 2. Batch Operations

```python
# Batch encrypt for better performance
vectors = [np.random.randn(512).astype(np.float32) for _ in range(100)]

# Single batch encrypt
ciphertexts = context.encrypt_batch(vectors)
```

### 3. Noise Budget

```python
# Monitor noise budget (CKKS has limited multiplication depth)
noise_budget = context.get_noise_budget(ciphertext)
print(f"Remaining noise budget: {noise_budget} bits")

# Relinearize after multiplications to manage noise
ct_relinearized = context.relinearize(ct_product)
```

### 4. Memory Management

```python
# Clear sensitive data when done
context.clear_secret_key()
del bundle.secret_key

# Force garbage collection
import gc
gc.collect()
```

## Troubleshooting

### Decryption Error

If decryption produces garbage:
- Noise budget exhausted (too many operations)
- Wrong keys used
- Corrupted ciphertext

```python
try:
    result = context.decrypt(ciphertext)
except DecryptionError as e:
    print(f"Decryption failed: {e}")
    # Check noise budget
    budget = context.get_noise_budget(ciphertext)
    print(f"Noise budget: {budget}")
```

### Performance Issues

```python
# Use smaller polynomial degree for faster (less secure) operations
fast_params = HESchemeParams(
    scheme=N2HEScheme.CKKS,
    poly_modulus_degree=4096,  # Faster but less secure
    security_level=128,
)
```

## Next Steps

- [Privacy Guide](../guides/privacy.md) - Full privacy features
- [LoRA Fine-Tuning](lora-finetuning.md) - Training with encryption
- [API Reference](../api-reference/configuration.md) - HE configuration

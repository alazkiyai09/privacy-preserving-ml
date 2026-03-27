# TEE ML Framework - User Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Basic Usage](#basic-usage)
5. [Advanced Usage](#advanced-usage)
6. [HT2ML Hybrid System](#ht2ml-hybrid-system)
7. [Security Best Practices](#security-best-practices)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

## Installation

### Requirements

- Python 3.8 or higher
- NumPy
- PyTest (for testing)

### Setup

```bash
# Navigate to project directory
cd /home/ubuntu/21Days_Project/tee_project

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tee_ml; print('TEE ML installed successfully')"

# Run tests
pytest tests/ -v
```

## Quick Start

### Your First TEE Program

```python
from tee_ml.core.enclave import create_enclave
import numpy as np

# Create an enclave
enclave = create_enclave(enclave_id="my-first-enclave")

# Prepare data
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Enter enclave and execute operation
session = enclave.enter(data)
result = session.execute(lambda arr: arr * 2)
enclave.exit(session)

print(f"Result: {result}")  # [2.0, 4.0, 6.0, 8.0, 10.0]
```

## Core Concepts

### What is a TEE?

A Trusted Execution Environment (TEE) is a secure area of a main processor. It guarantees code and data loaded inside to be protected with respect to confidentiality and integrity.

**Key Properties:**
- **Isolation**: Memory is isolated from the rest of the system
- **Integrity**: Code cannot be modified
- **Attestation**: Can prove what code is running
- **Sealed Storage**: Data can be encrypted to the specific enclave

### Enclave Lifecycle

```
Create → Enter → Execute → Exit → [Repeat]
  ↓        ↓       ↓       ↓
Init   Session  Result  Cleanup
```

### Session Management

A session represents an active enclave execution:
- Created when entering enclave with data
- Active while operations execute
- Closed when exiting enclave

## Basic Usage

### 1. Creating an Enclave

```python
from tee_ml.core.enclave import create_enclave

# Basic enclave
enclave = create_enclave(enclave_id="my-app")

# With memory limit
enclave = create_enclave(
    enclave_id="my-app",
    memory_limit_mb=256,
)
```

### 2. Executing Operations

```python
# Simple operation
data = np.array([1.0, 2.0, 3.0])
session = enclave.enter(data)

# Execute in TEE
result = session.execute(lambda arr: arr + 1)

# Exit
enclave.exit(session)
```

### 3. Using ML Operations

```python
from tee_ml.operations.activations import tee_relu, tee_softmax
from tee_ml.operations.comparisons import tee_argmax

# Activations
data = np.array([-1.0, 0.0, 1.0])
session = enclave.enter(data)

relu_result = tee_relu(data, session)  # [0.0, 0.0, 1.0]
softmax_result = tee_softmax(data, session)  # [0.09, 0.24, 0.67]
max_index = tee_argmax(data, session)  # 2

enclave.exit(session)
```

### 4. Remote Attestation

```python
from tee_ml.core.attestation import create_attestation_service

attestation = create_attestation_service(enclave)

# Generate attestation report
report = attestation.generate_report(
    enclave=enclave,
    nonce=b"client-challenge",
)

# Verify report
verification = attestation.verify_report(report)
print(f"Valid: {verification.is_valid}")
```

### 5. Sealed Storage

```python
from tee_ml.core.sealed_storage import create_sealed_storage

storage = create_sealed_storage(enclave)

# Seal sensitive data
secret = b"my-secret-key"
sealed = storage.seal(
    data=secret,
    enclave_id=enclave.enclave_id,
    measurement=enclave.get_measurement(),
)

# Unseal later
unsealed = storage.unseal(
    sealed_data=sealed,
    enclave_id=enclave.enclave_id,
    measurement=enclave.get_measurement(),
)
```

## Advanced Usage

### 1. Custom ML Layers

```python
from tee_ml.operations.activations import TeeActivationLayer

# Define custom activation
def custom_activation(x, session):
    return session.execute(lambda arr: np.where(arr > 0, arr, arr * 0.1))

# Use in network
layer = TeeActivationLayer(custom_activation)
result = layer.forward(data, session)
```

### 2. Batch Processing

```python
# Process multiple inputs
batch = np.random.randn(32, 10)  # 32 samples, 10 features
session = enclave.enter(batch)

def process_batch(arr):
    # Process each sample
    results = []
    for i in range(arr.shape[0]):
        sample = arr[i]
        # Process sample
        processed = sample * 2 + 1
        results.append(processed)
    return np.array(results)

results = session.execute(process_batch)
enclave.exit(session)
```

### 3. Error Handling

```python
try:
    session = enclave.enter(data)
    result = session.execute(operation)
    enclave.exit(session)
except MemoryError:
    print("Enclave memory limit exceeded")
except Exception as e:
    print(f"Enclave error: {e}")
    if session.is_active():
        enclave.exit(session)
```

### 4. Performance Monitoring

```python
# Get enclave statistics
stats = enclave.get_statistics()
print(f"Total sessions: {stats['total_sessions']}")
print(f"Memory used: {stats['memory_usage_mb']:.2f} MB")
print(f"Memory utilization: {stats['memory_utilization']:.2%}")
```

## HT2ML Hybrid System

### Overview

The HT2ML (Homomorphic Encryption + Trusted Execution) hybrid system combines:

1. **HE Layers** (1-2 layers): Cryptographic input privacy
2. **TEE Layers** (remaining): Hardware-protected computation
3. **Handoff Protocol**: Secure data transfer

### Finding Optimal Split

```python
from tee_ml.protocol.split_optimizer import (
    estimate_optimal_split,
    SplitStrategy,
)

# Get recommendation
recommendation = estimate_optimal_split(
    input_size=20,
    hidden_sizes=[10, 5],
    output_size=2,
    activations=['relu', 'sigmoid', 'softmax'],
    noise_budget=200,
    strategy=SplitStrategy.BALANCED,
)

# Display recommendation
recommendation.print_summary()
```

### Understanding Strategies

**PRIVACY_MAX**: Maximize HE layers for input privacy
- Use when: Input data is highly sensitive
- Trade-off: Slower performance

**PERFORMANCE_MAX**: Maximize TEE layers for speed
- Use when: Computation efficiency is critical
- Trade-off: Less input privacy

**BALANCED**: Optimal trade-off (recommended)
- Use when: Balance privacy and performance
- Trade-off: Moderate both

### HE↔TEE Handoff

```python
from tee_ml.protocol.handoff import (
    HEContext,
    HEData,
    create_handoff_protocol,
)

# Create protocol
protocol = create_handoff_protocol(enclave)

# Create HE context and encrypted data
he_context = HEContext(
    scheme='ckks',
    poly_modulus_degree=4096,
    scale=2**30,
    eval=1,
)

encrypted_data = HEData(
    encrypted_data=ciphertext_vector,
    shape=data.shape,
    scheme='ckks',
    scale=2**30,
)

# Perform handoff
success, plaintext = protocol.handoff_he_to_tee(
    encrypted_data=encrypted_data,
    he_context=he_context,
    nonce=b"fresh-nonce",
)

if success:
    # Process in TEE
    session = enclave.enter(plaintext)
    result = session.execute(operation)
    enclave.exit(session)
```

## Security Best Practices

### 1. Always Verify Attestation

```python
# Good: Verify before using
attestation = create_attestation_service(enclave)
report = attestation.generate_report(enclave, nonce=b"challenge")
verification = attestation.verify_report(report)

if verification.is_valid:
    # Use enclave
    pass
else:
    # Reject
    raise Exception("Attestation failed")
```

### 2. Use Constant-Time Operations

```python
from tee_ml.security.oblivious_ops import constant_time_eq

# Good: Constant-time comparison
if constant_time_eq(secret_a, secret_b):
    # Proceed
    pass

# Bad: Data-dependent comparison
if secret_a == secret_b:  # Leaks timing
    # Proceed
    pass
```

### 3. Validate Inputs

```python
def process_input(user_input, session):
    # Validate shape
    if user_input.shape != (10,):
        raise ValueError("Invalid input shape")

    # Validate range
    if np.any(np.abs(user_input) > 100):
        raise ValueError("Input out of range")

    # Process
    return session.execute(lambda arr: user_input * 2)
```

### 4. Use Nonces

```python
import os

# Always use fresh nonces
nonce = os.urandom(16)  # 128-bit nonce
success, plaintext = protocol.handoff_he_to_tee(
    encrypted_data=data,
    he_context=context,
    nonce=nonce,
)
```

### 5. Seal Sensitive Data

```python
# Good: Seal model weights
storage = create_sealed_storage(enclave)
sealed_weights = storage.seal(
    data=weights.tobytes(),
    enclave_id=enclave.enclave_id,
    measurement=enclave.get_measurement(),
)

# Never store plaintext secrets
# Bad: np.save("weights.npy", weights)
```

## Performance Optimization

### 1. Batch Operations

```python
# Bad: Multiple enclave entries
for item in items:
    session = enclave.enter(item)
    result = session.execute(operation)
    enclave.exit(session)

# Good: Single enclave entry
batch = np.array(items)
session = enclave.enter(batch)
results = session.execute(process_all_items)
enclave.exit(session)
```

### 2. Use Larger Data Sizes

```python
# Bad: Small data (high overhead)
small_data = np.array([1.0])

# Good: Larger data (amortize overhead)
large_data = np.random.randn(1000)
```

### 3. Minimize Cross-Enclave Calls

```python
# Bad: Multiple exits and entries
session1 = enclave.enter(data1)
result1 = session.execute(op1)
enclave.exit(session1)

session2 = enclave.enter(data2)
result2 = session.execute(op2)
enclave.exit(session2)

# Good: Single session
session = enclave.enter(data1)
result1 = session.execute(op1)
result2 = session.execute(op2)
enclave.exit(session)
```

### 4. Profile Before Optimizing

```python
from tee_ml.benchmarking import create_benchmark

benchmark = create_benchmark(enclave)

# Profile operation
result = benchmark.benchmark_tee_operation(
    operation=my_operation,
    data_size=1000,
    iterations=100,
)

print(f"Average time: {result.avg_time_ns / 1000:.2f} μs")
print(f"Throughput: {result.throughput_ops_per_sec:.2f} ops/sec")
```

## Troubleshooting

### Common Issues

**Issue: Memory limit exceeded**
```python
# Solution: Increase memory limit
enclave = create_enclave(enclave_id="my-app", memory_limit_mb=256)
```

**Issue: Slow performance**
```python
# Solution: Batch operations
batch = np.concatenate([item1, item2, item3])
session = enclave.enter(batch)
```

**Issue: Attestation fails**
```python
# Solution: Check measurement
expected_measurement = b"..."
if report.measurement != expected_measurement:
    raise Exception("Measurement mismatch")
```

### Debug Mode

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Enable debug logging
enclave = create_enclave(enclave_id="debug-app", debug=True)
```

## API Reference

### Core Classes

#### `Enclave`
Main enclave class for secure computation.

**Methods:**
- `__init__(enclave_id, memory_limit_mb=128)` - Create enclave
- `enter(data)` - Enter enclave with data
- `exit(session)` - Exit enclave
- `get_measurement()` - Get enclave measurement
- `get_statistics()` - Get usage statistics

#### `EnclaveSession`
Represents an active enclave session.

**Methods:**
- `execute(operation)` - Execute operation in enclave
- `get_memory_usage()` - Get memory usage
- `is_active()` - Check if active

### Operation Functions

#### Activations (`tee_ml.operations.activations`)
- `tee_relu(x, session)` - ReLU activation
- `tee_sigmoid(x, session)` - Sigmoid activation
- `tee_softmax(x, session, axis=-1)` - Softmax
- `tee_tanh(x, session)` - Hyperbolic tangent

#### Comparisons (`tee_ml.operations.comparisons`)
- `tee_argmax(x, session, axis=-1)` - Find max index
- `tee_threshold(x, threshold, session)` - Binary threshold
- `tee_top_k(x, k, session)` - Top-k elements
- `tee_where(condition, x, y, session)` - Conditional selection

#### Arithmetic (`tee_ml.operations.arithmetic`)
- `tee_divide(x, divisor, session)` - Division
- `tee_normalize(x, session, axis=-1)` - L2 normalization
- `tee_layer_normalization(x, gamma, beta, session)` - Layer norm
- `tee_batch_normalization(x, session)` - Batch norm

### Protocol Classes

#### `SplitOptimizer`
Analyzes optimal HE/TEE split point.

**Methods:**
- `recommend_split(layers, strategy)` - Get recommendation
- `find_feasible_splits(layers)` - Find all feasible splits
- `estimate_performance(layers, split_point)` - Estimate performance

#### `HT2MLProtocol`
Manages HE↔TEE handoff.

**Methods:**
- `handoff_he_to_tee(encrypted_data, he_context, nonce)` - HE to TEE handoff
- `handoff_tee_to_he(plaintext_data, he_context)` - TEE to HE handoff
- `get_handoff_statistics()` - Get statistics

### Benchmarking Classes

#### `TEEBenchmark`
Benchmarking framework.

**Methods:**
- `benchmark_function(func, iterations)` - Benchmark function
- `benchmark_tee_vs_plaintext(operation, tee_operation)` - Compare
- `benchmark_scalability(operation, data_sizes)` - Test scalability

#### `PerformanceReport`
Generate performance reports.

**Methods:**
- `add_benchmark_result(result)` - Add result
- `generate_summary(format)` - Generate report
- `save_report(filepath, format)` - Save to file

## Examples

See `examples/` directory for complete examples:
- `basic_usage.py` - Basic TEE operations
- `benchmarking_example.py` - Performance benchmarking
- `ht2ml_workflow.py` - Complete HT2ML workflow

## Support

For issues or questions:
1. Check examples in `examples/`
2. Review test cases in `tests/`
3. Read phase summaries in `PHASE*_SUMMARY.md`
4. Check documentation in `docs/`

## Further Reading

- [Intel SGX Documentation](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html)
- [HT2ML Paper](https://arxiv.org/abs/2305.06449)
- [Gramine Project](https://gramineproject.io/)

---

**Last Updated:** Phase 6 Complete ✅

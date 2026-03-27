# Privacy-Preserving ML Toolkit (`privacy-preserving-ml`)

Applied privacy and cryptography toolkit for ML systems, including **homomorphic encryption**, **trusted execution environments (TEE)**, **hybrid HE+TEE protocols**, **zero-knowledge verification**, and **encrypted gradient boosting**.

## Why This Repository

Production AI security requires layered privacy controls and verifiable computation. `privacy-preserving-ml` provides practical building blocks and experiment runners across multiple cryptographic approaches.

## Core Features

- HE modules (CKKS/BFV surfaces and benchmark hooks)
- TEE simulation and overhead modeling
- Hybrid protocol surfaces for HE+TEE execution
- ZKP prover/verifier and commitment verification modules
- Robust verification (Krum, median, trimmed mean variants)
- Encrypted GBDT training/secure split surfaces
- Experiment runners in `src/experiments`

## Project Structure

- `src/encryption/`: HE, TEE, and hybrid protocol modules
- `src/verification/`: ZKP, commitments, robust verification
- `src/models/encrypted_gbdt/`: encrypted tree-learning layers
- `src/experiments/`: runnable privacy/verification experiment entrypoints
- `src/core/`: errors, logging, types, validation, security utilities

## Quick Start

```bash
pip install -r requirements.txt
pytest -q tests/test_public_surfaces.py
```

## Experiment Runners

- `src/experiments/run_he_benchmark.py`
- `src/experiments/run_tee_benchmark.py`
- `src/experiments/run_hybrid_benchmark.py`
- `src/experiments/run_zkp_verification.py`
- `src/experiments/run_commitment_fl.py`
- `src/experiments/run_robust_fl.py`
- `src/experiments/run_encrypted_gbdt.py`

## SEO Keywords

privacy preserving machine learning, homomorphic encryption ml, tee machine learning, zk snark verification, commitment schemes federated learning, encrypted gbdt, secure ai model training

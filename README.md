<div align="center">

# 🔐 Privacy-Preserving ML

### Homomorphic Encryption • TEE • ZKP • Commitments • Encrypted GBDT

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![TenSEAL](https://img.shields.io/badge/TenSEAL-HE-5C2D91?style=flat)](https://github.com/OpenMined/TenSEAL)

[Overview](#-overview) • [About](#-about) • [Topics](#-topics) • [Quick Start](#-quick-start) • [Experiments](#-experiments)

---

Privacy and verification toolkit for ML systems combining **cryptography**, **trusted execution**, and **verifiable federated workflows**.

</div>

---

## 🎯 Overview

`privacy-preserving-ml` covers major privacy primitives:

- Homomorphic encryption (CKKS/BFV)
- TEE simulation and hybrid HE+TEE protocols
- ZK proof and commitment verification flows
- Robust federated verification and encrypted GBDT

## 📌 About

- Centralized repo for practical privacy-preserving ML components
- Built for experimentation with deployment-oriented module boundaries
- Includes benchmark and verification experiment runners

## 🏷️ Topics

`privacy-preserving-ml` `homomorphic-encryption` `tee` `zk-snark` `commitment-schemes` `encrypted-ml` `federated-learning` `cryptography`

## 🧩 Architecture

- `src/encryption/`: HE/TEE/hybrid protocol modules
- `src/verification/`: zkp, commitments, robust checks
- `src/models/encrypted_gbdt/`: encrypted tree-learning
- `src/experiments/`: runnable privacy and verification scripts
- `src/core/`: shared runtime and security utilities

## ⚡ Quick Start

```bash
pip install -r requirements.txt
pytest -q tests/test_public_surfaces.py
```

## 🧪 Experiments

- `src/experiments/run_he_benchmark.py`
- `src/experiments/run_tee_benchmark.py`
- `src/experiments/run_hybrid_benchmark.py`
- `src/experiments/run_zkp_verification.py`
- `src/experiments/run_commitment_fl.py`
- `src/experiments/run_robust_fl.py`
- `src/experiments/run_encrypted_gbdt.py`

## 🛠️ Tech Stack

**Crypto:** TenSEAL-style HE flows, commitment and proof systems  
**ML:** PyTorch + secure training components  
**Verification:** robust aggregation and proof validation modules

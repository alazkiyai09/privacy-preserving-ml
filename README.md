# privacy-preserving-ml

Privacy and cryptography toolkit for phishing-oriented ML workloads. This repo consolidates homomorphic encryption, TEE simulation, hybrid HE+TEE flows, verifiable learning primitives, robust aggregation, and encrypted GBDT building blocks.

## Layout

- `src/encryption/`: HE, TEE, and hybrid protocol surfaces
- `src/verification/`: ZKP, commitment, and robust verification modules
- `src/models/encrypted_gbdt/`: encrypted tree-learning helpers
- `src/core/`: embedded shared utilities
- `tests/`: lightweight public-surface smoke test plus preserved legacy suites

## Notes

- Original project trees are preserved under nested `legacy/` namespaces or root compatibility packages such as `he_ml/` and `tee_ml/`.
- The top-level public modules are thin, dependency-light facades meant to provide a stable structure for the split repo.

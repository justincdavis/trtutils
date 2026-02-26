# Docker CUDA Test Workflow

This directory contains local Docker tooling for running the full `trtutils` checks across
CUDA 11, 12, and 13.

## Prerequisites

The local runner (`ci/test_cuda.sh`) performs hard preflight checks and will fail if any are
missing or unhealthy:

- `docker`
- `docker compose` (v2)
- `nvidia-smi`
- `nvcc`

You also need a host with NVIDIA GPU support configured for Docker.

## Main Commands

From repo root:

- Build all project Docker images (`Dockerfile.act` + CUDA test images):
  - `make dockers`
- Run tests only:
  - `make test-cu11`
  - `make test-cu12`
  - `make test-cu13`
- Run full checks (lint + typecheck + tests) per CUDA:
  - `make ci-cu11`
  - `make ci-cu12`
  - `make ci-cu13`
- Run full checks across all CUDA versions:
  - `make ci-cu-all`

You can also run the script directly:

- `./ci/test_cuda.sh 11 --all`
- `./ci/test_cuda.sh 12 --all`
- `./ci/test_cuda.sh 13 --all`

## Pinned CUDA/TensorRT Requirements

Pinned Python package versions are defined per CUDA major:

- `docker/requirements_cu11.txt`
- `docker/requirements_cu12.txt`
- `docker/requirements_cu13.txt`

Each file pins:

- `cuda-python`
- `tensorrt_cu11` / `tensorrt_cu12` / `tensorrt_cu13`

`docker/docker-compose.test.yml` passes a per-service build arg (`CUDA_REQUIREMENTS_FILE`)
to `docker/Dockerfile.test`, which installs the selected requirements file during image build.

`pyproject.toml` stays intentionally loose; deterministic CUDA/TensorRT pinning is scoped to
these Docker test images.

## Updating Pinned Versions

1. Update the relevant `docker/requirements_cu*.txt` file(s).
2. Rebuild images:
   - `make dockers`
3. Re-run full checks:
   - `make ci-cu-all`

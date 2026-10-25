---
name: docker-ci
description: Use when testing trtutils across CUDA/TensorRT versions in Docker - "run the cu11/cu12/cu13 tests", "make test-cu12", "test in the CUDA container", "docker CI", "container can't see the GPU", "WSL2 cuInit fails in docker".
---

# CUDA 11/12/13 Docker test containers

`ci/test_cuda.sh` builds and runs `docker/docker-compose.test.yml` services `test-cu11`,
`test-cu12`, `test-cu13` (images `trtutils-test:cu1X`, built from `docker/Dockerfile.test`).

## Run

```bash
make test-cu12          # ./ci/test_cuda.sh 12 --test
make ci-cu12            # ./ci/test_cuda.sh 12 --all  (lint + typecheck + tests)
./ci/test_cuda.sh 13 --test --coverage
```

Flags: `--test`, `--lint`, `--typecheck`, `--coverage`, `--all`; no flag means lint +
typecheck + tests. Requires `docker compose` v2, the NVIDIA driver (`nvidia-smi`), and the
NVIDIA Container Toolkit. `make dockers` builds every project image (`docker/build.sh --all`).

## What the containers use

| Service | Base image (Docker Hub) | Requirements |
| --- | --- | --- |
| test-cu11 | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04` | `docker/requirements_cu11.txt` |
| test-cu12 | `nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04` | `docker/requirements_cu12.txt` |
| test-cu13 | `nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04` | `docker/requirements_cu13.txt` |

`src/`, `tests/`, `data/`, `ci/`, `pytest.ini`, `demos/`, `examples/` and the coverage files
are bind-mounted, so code edits need no rebuild; dependency changes do.

## Gotchas

- Use public `nvidia/cuda:*` images; NGC `nvcr.io/nvidia/tensorrt:*` needs a login.
- GPU access is via `deploy.resources.reservations.devices`, not `runtime: nvidia`.
- WSL2: the toolkit does not mount every driver library (`libnvdxgdmal.so.1`), so `cuInit`
  fails. `ci/test_cuda.sh` auto-detects WSL2 and adds `-v /usr/lib/wsl:/usr/lib/wsl:ro`;
  pass it yourself when calling `docker compose run` directly. `cuPvtCompilePtx` lines in
  `LD_DEBUG` output are a red herring.
- Ubuntu 24.04 (cu13) pip needs `--break-system-packages` (set via `PIP_EXTRA_FLAGS`).
- First builds are slow: the TensorRT libs wheels are several GB (~5 min in
  `Preparing metadata`).
- `pytest.ini` must be mounted for the custom markers to register.
- `.coverage.json` is `touch`ed first so Docker mounts a file, not a directory.
- Inside containers ruff/ty are installed system-wide; the `ci/run_*.sh` scripts find them
  via `command -v` when `.venv-ci` is absent.
- Engines are cached in the mounted `data/engines/` with the TensorRT version in the name,
  so containers and host do not clash.

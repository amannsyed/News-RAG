# Docker Images

This folder contains service-specific Dockerfiles and Python requirement files for the runtime containers.

## Files

- `Dockerfile.embedding`: GPU embedding API for `google/embeddinggemma-300m`.
- `Dockerfile.ner`: GPU NER API for `urchade/gliner_multi-v2.1`.
- `Dockerfile.rag-api`: FastAPI RAG/search/chat API container.
- `Dockerfile.mcp`: standalone FastMCP server container.
- `*-requirements.txt`: container-only dependencies kept separate from Poetry so the Docker images stay focused.

## Common Commands

Build and start the full local stack:

```bash
docker compose up -d --build
```

GPU model services require NVIDIA Docker support:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Check the resolved Compose configuration:

```bash
docker compose config
```

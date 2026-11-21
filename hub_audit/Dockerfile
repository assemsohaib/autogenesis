FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m auditor
USER auditor
WORKDIR /home/auditor
ENV UV_LINK_MODE=copy PATH=/home/auditor/.venv/bin:$PATH
RUN uv venv -q .venv -p 3.12 && uv pip install -q aiohttp

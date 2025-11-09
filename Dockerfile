FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        libclang-dev \
        llvm \
        git; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "src.server.rag_api:APP", "--host", "0.0.0.0", "--port", "8000"]

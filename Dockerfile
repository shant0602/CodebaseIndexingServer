FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        ca-certificates \
            gnupg \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libx11-dev \
        libxext-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev; \
    # Add LLVM 18 repository for compatibility with Python clang bindings
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm-snapshot.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg] http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-18 main" > /etc/apt/sources.list.d/llvm-18.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        clang-18 \
        libclang-18-dev \
        llvm-18 \
        llvm-18-dev; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "src.server.rag_api:APP", "--host", "0.0.0.0", "--port", "8000"]

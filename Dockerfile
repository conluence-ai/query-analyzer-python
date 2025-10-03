# Builder stage
FROM python:3.10-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install packages system-wide
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-slim
WORKDIR /app

# Copy everything from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

ENV PATH=/usr/local/bin:$PATH

EXPOSE 8432
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8432/health || exit 1

CMD ["gunicorn", "-w", "1", "--threads", "4", "--timeout", "120", "-b", "0.0.0.0:8432", "main:app"]

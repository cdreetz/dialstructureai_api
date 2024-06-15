# Use a Python slim image as a base
FROM python:3.12-slim as builder

# Set environment variables to minimize layer sizes and disable pip cache
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your Python application
COPY ./src ./src

# Remove build dependencies if not needed anymore
RUN apt-get purge -y --auto-remove gcc

# Second stage: Setup runtime environment
FROM python:3.12-slim

WORKDIR /app

# Copy from builder stage
COPY --from=builder /app /app

EXPOSE 80
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "80"]

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to minimize the image size and improve logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /usr/src/app

# Install necessary system packages for audio processing and compilation
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY src/ ./src/

# Run the Uvicorn server
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "-m", "src.app.main"]

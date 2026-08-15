# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies for image processing (OpenCV, SimpleITK)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install core Python libraries
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy your codebase into the container
COPY backend /app/backend

# Run the web service on container startup.
# The PORT environment variable is automatically injected by Cloud Run / Heroku / HF Spaces.
ENV PYTHONPATH="/app"
CMD uvicorn backend.oct_analyzer.api:app --host 0.0.0.0 --port ${PORT:-8000}

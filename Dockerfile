# Use a slim Python image
FROM python:3.10-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code and models
COPY backend/ ./backend/

# Expose the port FastAPI will run on
EXPOSE 7860

# Command to run the application
# Note: Hugging Face uses port 7860 by default
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]

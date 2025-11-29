FROM python:3.12-slim

# Environment
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    PORT=8501

# Install system dependencies needed by some packages (OpenCV, ffmpeg, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg curl \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt && \
    pip install onnx2tf

# Copy project files
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

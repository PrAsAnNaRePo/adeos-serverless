FROM python:3.10-slim-bullseye

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libcairo2-dev pkg-config build-essential ffmpeg libsm6 libxext6 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN pip install uv

# Install RunPod SDK first
RUN uv pip install --system runpod

# Install other dependencies
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
RUN uv pip install --system -U surya-ocr

# Copy all files
COPY . .

# Set timezone
RUN ln -snf /usr/share/zoneinfo/Asia/Kolkata /etc/localtime && echo "Asia/Kolkata" > /etc/timezone

# Start the RunPod handler
CMD ["python", "app.py"]
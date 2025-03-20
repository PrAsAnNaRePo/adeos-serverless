FROM python:3.10-slim-bullseye

RUN apt-get update && \
    apt-get install -y libcairo2-dev pkg-config build-essential ffmpeg libsm6 libxext6 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install RunPod SDK first
RUN pip install runpod

# Install other dependencies
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt
RUN python3.10 -m pip install -U surya-ocr

# Copy all files
COPY . .

# Set timezone
RUN ln -snf /usr/share/zoneinfo/Asia/Kolkata /etc/localtime && echo "Asia/Kolkata" > /etc/timezone

# Start the RunPod handler (not uvicorn)
CMD ["python", "app.py"]
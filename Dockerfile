FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y libcairo2-dev pkg-config build-essential ffmpeg libsm6 libxext6 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install runpod

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN ln -snf /usr/share/zoneinfo/Asia/Kolkata /etc/localtime && echo "Asia/Kolkata" > /etc/timezone

CMD ["python", "app.py"]
FROM arm64v8/python:3.10-slim-bullseye


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libgomp1 \
        libopenblas-base \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        python3-opencv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app/
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "run_detection_pipline.py"]
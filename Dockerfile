# CPU image. For NVIDIA GPUs build FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
# and install ".[gpu,server]" instead.
#
#   docker build -t drishti .
#   docker run -p 8000:8000 -e DRISHTI_API_KEY=change-me -v drishti-data:/data drishti
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 drishti \
    && mkdir -p /data && chown drishti:drishti /data
WORKDIR /app
COPY pyproject.toml README.md ./
COPY drishti ./drishti
RUN pip install --no-cache-dir ".[server,track]"
USER drishti
ENV DRISHTI__STORAGE__DB_PATH=/data/drishti.db \
    DRISHTI__STORAGE__CROPS_DIR=/data/crops \
    DRISHTI__IMAGING__BPC_MAP_DIR=/data/bad_pixels
RUN drishti models pull
VOLUME ["/data"]
EXPOSE 8000
# Set DRISHTI_API_KEY at run time: without it every /v1 endpoint is open to the network.
CMD ["drishti", "serve", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.8-slim-buster

WORKDIR /app/backend
COPY . .


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        build-essential \
        gcc \
        g++


RUN pip install --upgrade pip && \
    pip install --ignore-installed -r requirements.txt


RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


EXPOSE 8000
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" , "--reload"]

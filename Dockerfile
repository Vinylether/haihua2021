FROM huggingface/transformers-pytorch-gpu:latest
ENV DEBIAN_FRONTEND=noninteractive

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt


ENTRYPOINT ["python3", "main.py"]

FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY requirements.txt ./
RUN pip install --upgrade -r requirements.txt
RUN pip install flash_attn --no-build-isolation

RUN python -c "from transformers import AutoProcessor, AutoModelForImageTextToText; \
    AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-Instruct'); \
    AutoModelForImageTextToText.from_pretrained('HuggingFaceTB/SmolVLM-Instruct')"

COPY . .

CMD ["uvicorn", "src.app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
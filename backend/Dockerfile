# Production Dockerfile for PNG to Photoshop Layers Decomposer
FROM python:3.11-slim

# Prevent interactive prompts & configure Python
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DECOMPOSER_OCR_ENGINE=easyocr \
    DECOMPOSER_OCR_LANG=hi,en \
    DECOMPOSER_INPAINT_ENGINE=lama \
    DECOMPOSER_SEGMENTATION_ENGINE=yunet \
    DECOMPOSER_PSD_ENGINE=ag-psd

# Install system dependencies: Tesseract OCR (with English and Hindi), Cairo graphics, Pango
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    libgl1 \
    libglib2.0-0 \
    libcairo2-dev \
    libpango1.0-dev \
    libjpeg-dev \
    libgif-dev \
    librsvg2-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Node dependencies for ag-psd and canvas
COPY package.json package-lock.json ./
RUN npm ci

# Install PyTorch CPU first (saves ~3GB compared to default CUDA wheels)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache EasyOCR models (Hindi + English) so container starts instantly
RUN python -c "import easyocr; easyocr.Reader(['hi', 'en'], gpu=False)"

# Pre-download LaMa ONNX inpainting model
RUN mkdir -p /app/app/models && \
    curl -fL -o /app/app/models/lama_fp32.onnx https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx

# Copy application source code & fonts
COPY . .

# Ensure outputs directory exists
RUN mkdir -p /app/outputs

# Expose service port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

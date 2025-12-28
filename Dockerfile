FROM python:3.10-slim

# ----------------------------
# Environment settings
# ----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ----------------------------
# Working directory
# ----------------------------
WORKDIR /app

# ----------------------------
# System dependencies (required for LightGBM)
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Python dependencies
# ----------------------------
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ----------------------------
# Copy application code
# ----------------------------
COPY . .

# ----------------------------
# Expose FastAPI port
# ----------------------------
EXPOSE 8000

# ----------------------------
# Run FastAPI app
# ----------------------------
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]



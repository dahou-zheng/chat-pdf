# Use official Python slim image (Debian-based, smaller than full python)
FROM python:3.13-slim

# Set working directory inside container
WORKDIR /app

# Install minimal system dependencies needed for:
# - PyMuPDF (MuPDF C libs, fonts, rendering)
# - numpy / faiss-cpu (basic linear algebra, OpenBLAS fallback if needed)
# - General build tools (in case any dep needs compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip in container
RUN pip install --upgrade pip

# Copy requirements first → better Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
# --server.port=8501 --server.address=0.0.0.0 is standard for Docker
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
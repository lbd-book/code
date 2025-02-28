# Use a base image with Python pre-installed
FROM python:3.11.9-slim-bullseye

# Install JupyterLab and related dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    libsqlite3-dev \
    libbz2-dev \
    libreadline-dev \
    liblzma-dev \
    zlib1g-dev && \
    python3 -m pip install --upgrade pip && \
    pip install jupyterlab && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Ensure Jupyter runs properly
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

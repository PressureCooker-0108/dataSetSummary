# Use a stable, lightweight official Python runtime as base
FROM python:3.11-slim

# Set system-wide environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up the application workspace directory
WORKDIR /app

# Create a non-privileged user and setup directories
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Docker image user" appuser

# Copy dependency specifications and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set correct ownership of directories
RUN mkdir -p logs reports data && chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to verify service availability on dynamic port
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD sh -c "curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1"

# Launch the Streamlit application dynamically binding to Railway's $PORT
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]

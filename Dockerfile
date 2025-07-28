FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files including tickers.json
COPY app.py reddit_pipeline.py scheduler.py ./
COPY tickers.json ./
COPY start.sh ./

# Copy streamlit config if it exists
COPY .streamlit/ ./.streamlit/ 2>/dev/null || true

# Create necessary directories
RUN mkdir -p logs data

# Expose port for Streamlit
EXPOSE 8501

# Create startup script
COPY start.sh .
RUN chmod +x start.sh

# Run the application
CMD ["./start.sh"]
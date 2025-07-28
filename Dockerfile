FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py reddit_pipeline.py scheduler.py start.sh tickers.json ./
RUN mkdir -p logs data && chmod +x start.sh

EXPOSE 8501
CMD ["./start.sh"]
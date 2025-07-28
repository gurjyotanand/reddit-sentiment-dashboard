#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data logs

# Set environment variables for Streamlit
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Start Streamlit
echo "Starting Reddit Stock Sentiment Dashboard..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
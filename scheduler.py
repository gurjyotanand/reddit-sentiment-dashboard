import schedule
import time
import threading
import json
import os
from datetime import datetime
from reddit_pipeline import main as run_pipeline
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def update_data():
    """Run the Reddit pipeline and update timestamp"""
    try:
        logger.info("Starting scheduled data update...")
        
        # Run the pipeline
        run_pipeline()
        
        # Update the last updated timestamp
        timestamp_data = {
            "last_updated": datetime.now().isoformat(),
            "status": "success"
        }
        
        with open('data/last_updated.json', 'w') as f:
            json.dump(timestamp_data, f)
            
        logger.info("Data update completed successfully")
        
    except Exception as e:
        logger.error(f"Error during scheduled update: {str(e)}")
        
        # Update timestamp with error status
        timestamp_data = {
            "last_updated": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }
        
        with open('data/last_updated.json', 'w') as f:
            json.dump(timestamp_data, f)

def run_scheduler():
    """Run the scheduler in a separate thread"""
    # Schedule the job every 10 minutes
    schedule.every(10).minutes.do(update_data)
    
    # Run once immediately on startup
    update_data()
    
    logger.info("Scheduler started - running every 10 minutes")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

def start_scheduler():
    """Start the scheduler in a background thread"""
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    return scheduler_thread

if __name__ == "__main__":
    run_scheduler()
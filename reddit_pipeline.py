# reddit_pipeline.py
# Combines Reddit scraping and sentiment analysis into a single workflow.

# --- Combined Imports from both scripts ---
import praw
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os
import logging
from typing import List, Dict, Any, Tuple
import reticker
import re
import sys
import subprocess
import importlib.util
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Dependency Management from sentiment_analyzer.py ---

def install_package(package_name: str):
    """Install package if not available"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package_name}")
        raise

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = {
        'vaderSentiment': 'vaderSentiment'
        # praw, pandas, and reticker are assumed to be installed as per the original scraper
    }
    
    for import_name, package_name in required_packages.items():
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            print(f"Dependency '{package_name}' not found. Installing...")
            install_package(package_name)

# --- Check dependencies at the start ---
check_and_install_dependencies()

# Now we can safely import vader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# --- Class definition from reddit_scrapper.py ---

class RedditScraper:
    def __init__(self, client_id: str, client_secret: str, user_agent: str, 
                 min_comment_karma: int = 100, min_account_age_days: int = 30,
                 tickers_file: str = "tickers.json"):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.min_comment_karma = min_comment_karma
        self.min_account_age_days = min_account_age_days
        self.ticker_extractor = reticker.TickerExtractor()
        self.valid_tickers = self.load_valid_tickers(tickers_file)
        
        # EXPANDED excluded tickers list - common English words that get misidentified
        self.excluded_tickers = {
            'A', 'I', 'GO', 'ON', 'IT', 'BE', 'DD', 'CEO', 'PR', 'USA', 'FOR', 
            'NOW', 'YOLO', 'THE', 'GAIN', 'LOSS', 'EPS', 'PE', 'BUY', 'SELL', 
            'HOLD', 'ALL', 'ARE', 'CAN', 'BIG', 'TOP', 'EOD', 'PM', 'AH',
            # Additional common words causing false positives
            'YOU', 'UP', 'OUT', 'SO', 'OR', 'GOOD', 'AN', 'HAS', 'BY', 'AS', 
            'SEE', 'BACK', 'AT', 'TO', 'IN', 'OF', 'AND', 'IS', 'IF', 'BUT',
            'NOT', 'GET', 'MY', 'NO', 'YES', 'DO', 'NEW', 'OLD', 'WAY', 'USE',
            'ONE', 'TWO', 'ANY', 'WHO', 'WHY', 'HOW', 'LOW', 'HIGH', 'BAD',
            'GOOD', 'BEST', 'LAST', 'NEXT', 'ONLY', 'MAIN', 'REAL', 'FULL',
            'SURE', 'SAME', 'LONG', 'WELL', 'MUCH', 'MOST', 'MANY', 'SOME',
            'VERY', 'JUST', 'EVEN', 'ALSO', 'BOTH', 'EACH', 'SUCH', 'THAN',
            'THEY', 'THEM', 'THEIR', 'THERE', 'THEN', 'WHEN', 'WHERE', 'WHAT',
            'WHICH', 'WHILE', 'AFTER', 'BEFORE', 'AGAIN', 'STILL', 'SINCE',
            'UNTIL', 'ALMOST', 'ALWAYS', 'NEVER', 'OFTEN', 'USUALLY', 'SOMETIMES'
        }
        self.setup_logging()
        
    def load_valid_tickers(self, tickers_file: str) -> set:
        """
        Load valid tickers from JSON file. Exits the program if the file is not found or invalid.
        """
        if not os.path.exists(tickers_file):
            print(f"FATAL ERROR: Ticker file '{tickers_file}' not found.")
            print("The program cannot continue without a valid list of tickers for validation.")
            sys.exit(1)
            
        try:
            with open(tickers_file, 'r', encoding='utf-8') as f:
                ticker_data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(ticker_data, list):
                # Flat list: ["AAPL", "GOOG", ...]
                valid_tickers = {ticker.upper() for ticker in ticker_data if isinstance(ticker, str) and len(ticker) >= 1}
            elif isinstance(ticker_data, dict):
                # Could be {"tickers": ["AAPL", ...]} or {"AAPL": {...}, "GOOG": {...}}
                if "tickers" in ticker_data and isinstance(ticker_data["tickers"], list):
                    valid_tickers = {ticker.upper() for ticker in ticker_data["tickers"] if isinstance(ticker, str) and len(ticker) >= 1}
                else:
                    # Assume keys are tickers
                    valid_tickers = {ticker.upper() for ticker in ticker_data.keys() if isinstance(ticker, str) and len(ticker) >= 1}
            else:
                print(f"FATAL ERROR: Ticker file '{tickers_file}' has unsupported format.")
                sys.exit(1)
                
            if not valid_tickers:
                print(f"WARNING: No valid tickers were loaded from {tickers_file}. It might be empty.")
            else:
                print(f"Successfully loaded {len(valid_tickers)} unique tickers from {tickers_file}")
                # Debug: Show first 10 tickers to verify loading
                sample_tickers = list(valid_tickers)[:10]
                print(f"Sample tickers loaded: {', '.join(sample_tickers)}")
            
            return valid_tickers
            
        except json.JSONDecodeError:
            print(f"FATAL ERROR: Invalid JSON format in '{tickers_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"FATAL ERROR: An unexpected error occurred while loading tickers from {tickers_file}: {str(e)}")
            sys.exit(1)

    def extract_and_validate_tickers(self, text: str) -> List[str]:
        """
        Extracts tickers from text and VALIDATES them against the loaded ticker list.
        Only returns tickers that exist in the tickers.json file.
        
        Args:
            text: Text to extract tickers from
            
        Returns:
            List of valid ticker symbols found in the text, ONLY validated symbols.
        """
        if not text:
            return []
        
        try:
            # Step 1: Extract potential tickers from the text
            extracted_tickers = self.ticker_extractor.extract(text)
            
            # Step 2: Filter out common words and validate against ticker list
            validated_tickers = []
            
            for ticker in extracted_tickers:
                ticker = ticker.upper().strip()
                
                # Skip empty tickers
                if not ticker:
                    continue
                
                # Skip single characters (too likely to be false positives)
                if len(ticker) < 2:
                    continue
                
                # Skip common English words
                if ticker in self.excluded_tickers:
                    self.logger.debug(f"Excluded common word: {ticker}")
                    continue
                
                # MOST IMPORTANT: Only include if it exists in our valid tickers list
                if ticker not in self.valid_tickers:
                    self.logger.debug(f"Ticker '{ticker}' not found in valid tickers list")
                    continue
                
                validated_tickers.append(ticker)
            
            # Step 3: Remove duplicates while preserving order
            seen = set()
            unique_validated_tickers = []
            for ticker in validated_tickers:
                if ticker not in seen:
                    seen.add(ticker)
                    unique_validated_tickers.append(ticker)
            
            if unique_validated_tickers:
                self.logger.debug(f"Validated tickers found: {', '.join(unique_validated_tickers)}")
            
            return unique_validated_tickers
            
        except Exception as e:
            self.logger.warning(f"Error extracting tickers from text '{text[:50]}...': {str(e)}")
            return []
        
    def setup_logging(self):
        if not os.path.exists('logs'): 
            os.makedirs('logs')
        log_filename = f"logs/reddit_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'), 
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Reddit scraper initialized.")

    def get_user_info(self, username: str) -> Dict[str, Any]:
        try:
            if username == '[deleted]': 
                return {'username': '[deleted]', 'comment_karma': 0, 'account_age_days': 0, 'exists': False}
            user = self.reddit.redditor(username)
            created_utc = user.created_utc
            account_created = datetime.fromtimestamp(created_utc)
            return {
                'username': username, 
                'comment_karma': user.comment_karma, 
                'link_karma': user.link_karma,
                'account_age_days': (datetime.now() - account_created).days, 
                'exists': True
            }
        except Exception:
            return {'username': username, 'comment_karma': 0, 'account_age_days': 0, 'exists': False}

    def should_filter_comment(self, user_info: Dict[str, Any]) -> bool:
        if not user_info['exists']: 
            return True
        if user_info['comment_karma'] < self.min_comment_karma: 
            return True
        if user_info['account_age_days'] < self.min_account_age_days: 
            return True
        return False

    def get_latest_lounge_thread(self, subreddit_name: str) -> Dict[str, Any]:
        subreddit = self.reddit.subreddit(subreddit_name)
        for submission in subreddit.search("The Lounge", sort='new', limit=10):
            if "lounge" in submission.title.lower():
                return {
                    'id': submission.id, 
                    'title': submission.title, 
                    'author': str(submission.author) if submission.author else '[deleted]',
                    'num_comments': submission.num_comments, 
                    'created_utc': datetime.fromtimestamp(submission.created_utc),
                    'url': submission.url
                }
        return None
    
    def get_all_thread_comments(self, thread_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        submission = self.reddit.submission(id=thread_id)
        submission.comment_sort = 'new'
        submission.comments.replace_more(limit=None)
        filtered_comments, filtered_out_comments = [], []
        
        self.logger.info(f"Processing {len(submission.comments.list())} comments from thread: {submission.title}")
        
        for i, comment in enumerate(submission.comments.list(), 1):
            if i % 100 == 0: 
                self.logger.info(f"Processing comment {i}/{len(submission.comments.list())}")
            
            author_name = str(comment.author) if comment.author else '[deleted]'
            user_info = self.get_user_info(author_name)
            extracted_tickers = self.extract_and_validate_tickers(comment.body)
            
            comment_data = {
                'id': comment.id, 
                'body': comment.body, 
                'author': author_name, 
                'score': comment.score,
                'created_utc': datetime.fromtimestamp(comment.created_utc), 
                'permalink': comment.permalink,
                'author_comment_karma': user_info['comment_karma'], 
                'author_account_age_days': user_info['account_age_days'],
                'tickers': ', '.join(extracted_tickers) if extracted_tickers else '', 
                'ticker_count': len(extracted_tickers)
            }
            
            if self.should_filter_comment(user_info):
                filtered_out_comments.append(comment_data)
            else:
                filtered_comments.append(comment_data)
            
            #time.sleep(0.05)  # Small delay to be nice to Reddit's API
        
        self.logger.info(f"Finished processing. Kept: {len(filtered_comments)}, Filtered out: {len(filtered_out_comments)}")
        return filtered_comments, filtered_out_comments

    def save_to_json(self, data: List[Dict[str, Any]], filename: str):
        json_data = []
        for item in data:
            json_item = item.copy()
            for key, value in json_item.items():
                if isinstance(value, datetime):
                    json_item[key] = value.isoformat()
            json_data.append(json_item)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Data saved to {filename}")


# --- Class definition from reddit_sentiment_analyzer.py ---

class RedditSentimentEnhancer:
    def __init__(self, sentiment_threshold: float = 0.15):
        self.analyzer = SentimentIntensityAnalyzer()
        self.threshold = sentiment_threshold
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.user_pattern = re.compile(r'/?u/[A-Za-z0-9_-]+')
        self.sub_pattern = re.compile(r'/?r/[A-Za-z0-9_-]+')
        self.excess_punct = re.compile(r'[^\w\s.,!?$%-]')
        self.whitespace = re.compile(r'\s+')
    
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip(): 
            return ""
        cleaned = text.lower()
        cleaned = self.url_pattern.sub(' ', cleaned)
        cleaned = self.user_pattern.sub(' ', cleaned)
        cleaned = self.sub_pattern.sub(' ', cleaned)
        cleaned = self.excess_punct.sub(' ', cleaned)
        cleaned = self.whitespace.sub(' ', cleaned)
        return cleaned.strip()
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return {
                'sentiment': 'neutral', 
                'sentiment_compound': 0.0, 
                'sentiment_confidence': 0.0,
                'sentiment_positive': 0.0, 
                'sentiment_negative': 0.0, 
                'sentiment_neutral': 1.0,
                'cleaned_text': cleaned_text
            }
        
        scores = self.analyzer.polarity_scores(cleaned_text)
        compound = scores['compound']
        
        if compound >= self.threshold: 
            sentiment = 'bullish'
        elif compound <= -self.threshold: 
            sentiment = 'bearish'
        else: 
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment, 
            'sentiment_compound': round(compound, 4),
            'sentiment_confidence': round(abs(compound), 4), 
            'sentiment_positive': round(scores['pos'], 4),
            'sentiment_negative': round(scores['neg'], 4), 
            'sentiment_neutral': round(scores['neu'], 4),
            'cleaned_text': cleaned_text
        }
    
    def enhance_comment(self, comment: Dict[str, Any]) -> Dict[str, Any]:
        body = comment.get('body', '')
        sentiment_data = self.analyze_sentiment(body)
        enhanced_comment = comment.copy()
        enhanced_comment.update(sentiment_data)
        return enhanced_comment
    
    def enhance_comments(self, comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print(f"\nPerforming sentiment analysis on {len(comments)} comments...")
        enhanced_comments = []
        for i, comment in enumerate(comments):
            enhanced_comments.append(self.enhance_comment(comment))
            if (i + 1) % 100 == 0 or i == len(comments) - 1:
                print(f"  Processed {i + 1}/{len(comments)} comments", end='\r')
        print("\nSentiment analysis complete.")
        return enhanced_comments
    
    def get_sentiment_summary(self, enhanced_comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_comments = len(enhanced_comments)
        if total_comments == 0: 
            return {'total_comments': 0}
        
        sentiment_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        ticker_sentiment = {}
        
        for comment in enhanced_comments:
            sentiment = comment.get('sentiment', 'neutral')
            sentiment_counts[sentiment] += 1
            tickers_str = comment.get('tickers', '')
            if tickers_str:
                tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
                for ticker in tickers:
                    if ticker not in ticker_sentiment:
                        ticker_sentiment[ticker] = {'bullish': 0, 'bearish': 0, 'neutral': 0, 'total': 0}
                    ticker_sentiment[ticker][sentiment] += 1
                    ticker_sentiment[ticker]['total'] += 1

        sentiment_percentages = {s: round((c / total_comments) * 100, 2) for s, c in sentiment_counts.items()}
        return {
            'total_comments': total_comments, 
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': sentiment_percentages, 
            'ticker_sentiment': ticker_sentiment
        }

# --- NEW Integrated Main Function ---

def main():
    """
    Main function to run the full Reddit scraping and sentiment analysis pipeline.
    """
    # --- Step 1: Configuration from Environment Variables ---
    CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "Wcbrp8MCqzjEC1numM_U_w")
    CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "zgP5AmvgRTWwH18SNWbR2jqHuxcb_w")
    USER_AGENT = os.getenv("REDDIT_USER_AGENT", "RedditPipeline/1.0 (by /u/your_username)")
    
    MIN_COMMENT_KARMA = int(os.getenv("MIN_COMMENT_KARMA", "100"))
    MIN_ACCOUNT_AGE_DAYS = int(os.getenv("MIN_ACCOUNT_AGE_DAYS", "30"))
    SENTIMENT_THRESHOLD = float(os.getenv("SENTIMENT_THRESHOLD", "0.15"))
    
    # Initialize the scraper
    scraper = RedditScraper(
        CLIENT_ID, 
        CLIENT_SECRET, 
        USER_AGENT,
        min_comment_karma=MIN_COMMENT_KARMA,
        min_account_age_days=MIN_ACCOUNT_AGE_DAYS,
        tickers_file="tickers.json"
    )
    
    # --- Step 2: Scrape Reddit Data ---
    print("Searching for the latest 'The Lounge' thread in r/pennystocks...")
    lounge_thread = scraper.get_latest_lounge_thread("pennystocks")
    
    if not lounge_thread:
        print("No 'The Lounge' thread found. Exiting.")
        return

    print(f"Found thread: '{lounge_thread['title']}' with {lounge_thread['num_comments']} comments.")
    print(f"Fetching and filtering comments...")
        
    filtered_comments, _ = scraper.get_all_thread_comments(lounge_thread['id'])
        
    if not filtered_comments:
        print("No comments remained after filtering. Exiting.")
        return

    # --- Step 3: Perform Sentiment Analysis (In-Memory) ---
    enhancer = RedditSentimentEnhancer(sentiment_threshold=SENTIMENT_THRESHOLD)
    enhanced_comments = enhancer.enhance_comments(filtered_comments)
    
    # --- Step 4: Save Final Output ---
    output_filename = "lounge_thread_with_sentiment.json"
    scraper.save_to_json(enhanced_comments, output_filename)

    # --- Step 5: Display Summary ---
    summary = enhancer.get_sentiment_summary(enhanced_comments)
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE - SENTIMENT ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total comments analyzed: {summary['total_comments']}")
    print(f"Bullish: {summary['sentiment_counts']['bullish']} ({summary['sentiment_percentages']['bullish']}%)")
    print(f"Bearish: {summary['sentiment_counts']['bearish']} ({summary['sentiment_percentages']['bearish']}%)")
    print(f"Neutral: {summary['sentiment_counts']['neutral']} ({summary['sentiment_percentages']['neutral']}%)")
    
    if summary['ticker_sentiment']:
        print("\n--- Top Ticker Sentiments ---")
        # Sort tickers by total mentions
        sorted_tickers = sorted(summary['ticker_sentiment'].items(), key=lambda item: item[1]['total'], reverse=True)
        for ticker, counts in sorted_tickers[:15]:  # Show top 15
            print(f"  ${ticker} (Total: {counts['total']}): {counts['bullish']} Bullish, {counts['bearish']} Bearish, {counts['neutral']} Neutral")
    else:
        print("\n--- No valid tickers found in comments ---")
        print("This could mean:")
        print("1. No actual stock tickers were mentioned")
        print("2. The tickers.json file doesn't contain the mentioned tickers")
        print("3. All mentions were filtered out as common words")

    print(f"\nSuccess! Full enriched data saved to: {output_filename}")


if __name__ == "__main__":
    # Ensure you have a 'tickers.json' file in the same directory.
    if not os.path.exists('tickers.json'):
        print("ERROR: 'tickers.json' not found. Please create it.")
        print('Example format: ["AAPL", "GOOG", "TSLA"] or {"tickers": ["AAPL", "GOOG", "TSLA"]}')
    else:
        main()
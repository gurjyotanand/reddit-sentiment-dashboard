import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from scheduler import start_scheduler

# --- Page Configuration ---
st.set_page_config(
    page_title="Reddit Stock Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto"
)

# Start the scheduler when the app starts
if 'scheduler_started' not in st.session_state:
    start_scheduler()
    st.session_state.scheduler_started = True

# --- Helper Functions ---
def get_last_updated():
    """Get the last updated timestamp"""
    try:
        if os.path.exists('data/last_updated.json'):
            with open('data/last_updated.json', 'r') as f:
                data = json.load(f)
                return data.get('last_updated'), data.get('status', 'unknown')
        return None, 'unknown'
    except:
        return None, 'error'

def format_timestamp(timestamp_str):
    """Format timestamp for display"""
    if not timestamp_str:
        return "Never updated"
    
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except:
        return "Invalid timestamp"

# --- Data Loading and Caching ---
@st.cache_data(ttl=60)  # Cache for 1 minute
def load_data(file_path):
    """Loads and preprocesses the reddit data from a JSON file."""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return pd.DataFrame(), "File not found"
        
        # Load the data from the JSON file
        df = pd.read_json(file_path)
        
        if df.empty:
            return pd.DataFrame(), "File is empty"

        # --- Data Cleaning & Transformation ---
        # Ensure 'tickers' is a string and handle empty values
        df['tickers'] = df['tickers'].fillna('').astype(str)
        
        # Convert UTC timestamp to a readable datetime object
        df['created_datetime'] = pd.to_datetime(df['created_utc'])
        
        # Split the 'tickers' string into a list of actual tickers
        df['ticker_list'] = df['tickers'].apply(lambda x: [ticker.strip() for ticker in x.split(',') if ticker.strip()])
        
        return df, "success"
        
    except FileNotFoundError:
        return pd.DataFrame(), "File not found"
    except json.JSONDecodeError:
        return pd.DataFrame(), "Invalid JSON format"
    except Exception as e:
        return pd.DataFrame(), f"Error: {str(e)}"

# --- Main Application UI ---
st.title("📈 Reddit PennyStocks Sentiment Dashboard")
st.markdown("Analyzing comments from Reddit to gauge market sentiment on various stocks.")

# Display last updated info
last_updated, status = get_last_updated()
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if last_updated:
        if status == "success":
            st.success(f"📊 Last updated: {format_timestamp(last_updated)}")
        elif status == "error":
            st.error(f"⚠️ Last update failed: {format_timestamp(last_updated)}")
        else:
            st.info(f"🔄 Status unknown: {format_timestamp(last_updated)}")
    else:
        st.warning("🕐 No update information available")

with col2:
    if st.button("🔄 Refresh Data", key="refresh"):
        st.cache_data.clear()
        st.rerun()

with col3:
    st.caption("Updates every 10 minutes")

# Load the data
df_main, load_status = load_data('lounge_thread_with_sentiment.json')

if load_status != "success":
    st.error(f"Failed to load data: {load_status}")
    st.info("The data pipeline might be running for the first time. Please wait a few minutes and refresh.")
    st.stop()

if df_main.empty:
    st.warning("No data available. The scraper might be running for the first time.")
    st.info("Please wait for the next update cycle (up to 10 minutes) and refresh the page.")
    st.stop()

# Display data summary
st.info(f"📈 Showing {len(df_main)} comments from the latest analysis")

# Explode the dataframe to have one row per ticker mention for easier analysis
df_exploded = df_main.explode('ticker_list').rename(columns={'ticker_list': 'ticker'}).dropna(subset=['ticker'])

# --- Feature 1: Top 10 Tickers by Mentions ---
st.header("1. Top 10 Tickers by Mentions")

if not df_exploded.empty:
    # Create sentiment columns for easy aggregation
    df_exploded['bullish'] = (df_exploded['sentiment'] == 'bullish').astype(int)
    df_exploded['bearish'] = (df_exploded['sentiment'] == 'bearish').astype(int)
    df_exploded['neutral'] = (df_exploded['sentiment'] == 'neutral').astype(int)

    # Group by ticker and aggregate the data
    top_tickers = df_exploded.groupby('ticker').agg(
        mentions=('ticker', 'count'),
        upvotes=('score', 'sum'),
        bullish=('bullish', 'sum'),
        bearish=('bearish', 'sum'),
        neutral=('neutral', 'sum')
    ).sort_values(by='mentions', ascending=False).reset_index()

    # Add sentiment percentages
    top_tickers['bullish_pct'] = (top_tickers['bullish'] / top_tickers['mentions'] * 100).round(1)
    top_tickers['bearish_pct'] = (top_tickers['bearish'] / top_tickers['mentions'] * 100).round(1)
    
    st.dataframe(top_tickers.head(10), use_container_width=True)
else:
    st.warning("No ticker mentions found in the data.")

# --- Feature 2: Watchlist (Comments with 3+ Tickers) ---
st.header("2. High-Diversity Watchlist")
st.markdown("Comments mentioning three or more tickers, potentially indicating broader market discussion.")

watchlist_df = df_main[df_main['ticker_count'] >= 3][['tickers', 'score', 'body', 'sentiment']].sort_values(by='score', ascending=False)

if not watchlist_df.empty:
    st.dataframe(watchlist_df, use_container_width=True)
else:
    st.info("No comments found that mention 3 or more tickers.")

# --- Feature 3: Top 10 Most Upvoted Comments ---
st.header("3. Top 10 Upvoted Comments")
st.markdown("The most popular comments, which can heavily influence opinion.")

top_comments_df = df_main.sort_values(by='score', ascending=False).head(10)

for index, row in top_comments_df.iterrows():
    # Add sentiment emoji
    sentiment_emoji = "🐂" if row['sentiment'] == 'bullish' else "🐻" if row['sentiment'] == 'bearish' else "😐"
    
    st.markdown(f"> {row['body']}")
    st.markdown(f"""
    **Score:** {row['score']} | **Tickers:** `{row['tickers'] if row['tickers'] else 'N/A'}` | 
    **Author:** u/{row['author']} | **Sentiment:** {row['sentiment']} {sentiment_emoji}
    """)
    st.divider()

# --- Feature 4: Latest Comments for Top 5 Tickers ---
st.header("4. Latest Comments for Top Tickers")
st.markdown("Track the most recent discussions for the most mentioned stocks.")

if not df_exploded.empty:
    top_5_ticker_list = top_tickers['ticker'].head(5).tolist()

    for ticker in top_5_ticker_list:
        st.subheader(f"Latest Comments for `${ticker}`")
        
        # Filter for comments mentioning the current ticker
        latest_comments = df_exploded[df_exploded['ticker'] == ticker].sort_values(by='created_datetime', ascending=False).head(5)
        
        if latest_comments.empty:
            st.write(f"No comments found for ${ticker}.")
            continue

        for _, comment in latest_comments.iterrows():
            sentiment_emoji = "🐂" if comment['sentiment'] == 'bullish' else "🐻" if comment['sentiment'] == 'bearish' else "😐"
            confidence = comment.get('sentiment_confidence', 0)
            
            st.markdown(
                f"""
                <div style="border-left: 5px solid #ccc; padding-left: 10px; margin-bottom: 10px;">
                    <p>{comment['body']}</p>
                    <small><b>Author:</b> u/{comment['author']} | <b>Score:</b> {comment['score']} | 
                    <b>Sentiment:</b> {comment['sentiment']} {sentiment_emoji} (Confidence: {confidence:.2f})</small>
                </div>
                """, 
                unsafe_allow_html=True
            )
else:
    st.warning("Cannot generate latest comments as no tickers were found.")

# --- Feature 5: Overall Sentiment Summary ---
st.header("5. Overall Sentiment Summary")

sentiment_summary = df_main['sentiment'].value_counts()
sentiment_pct = (sentiment_summary / len(df_main) * 100).round(1)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🐂 Bullish", f"{sentiment_summary.get('bullish', 0)}", f"{sentiment_pct.get('bullish', 0)}%")
with col2:
    st.metric("🐻 Bearish", f"{sentiment_summary.get('bearish', 0)}", f"{sentiment_pct.get('bearish', 0)}%")
with col3:
    st.metric("😐 Neutral", f"{sentiment_summary.get('neutral', 0)}", f"{sentiment_pct.get('neutral', 0)}%")

# Footer
st.markdown("---")
st.markdown("💡 **Data automatically updates every 10 minutes** | Built with Streamlit & Reddit API")
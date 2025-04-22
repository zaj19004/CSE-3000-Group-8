"""
Bias Analysis of Political News on CNN, Fox News, and The New York Times

This script demonstrates how to:
1. Fetch political news articles via RSS feeds.
2. Extract article text using BeautifulSoup instead of `newspaper3k`.
3. Perform sentiment analysis with NLTK's VADER as a proxy for bias.
4. Visualize sentiment distributions with matplotlib.

Required:
    pip install feedparser nltk pandas matplotlib beautifulsoup4 requests
"""

import feedparser
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Define RSS feed URLs
feeds = {
    'CNN': 'http://rss.cnn.com/rss/cnn_allpolitics.rss',
    'Fox News': 'http://feeds.foxnews.com/foxnews/politics',
    'NYT': 'https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml'
}

# Step 2: Scrape article content using BeautifulSoup
def fetch_articles(feed_url, max_articles=10):
    feed = feedparser.parse(feed_url)
    articles = []
    for entry in feed.entries[:max_articles]:
        try:
            res = requests.get(entry.link, timeout=10)
            soup = BeautifulSoup(res.content, 'html.parser')
            # Grab all paragraphs as text
            paragraphs = soup.find_all('p')
            text = ' '.join(p.get_text() for p in paragraphs)
            articles.append({
                'title': entry.title,
                'text': text,
                'published': entry.published
            })
        except Exception as e:
            print(f"Failed to fetch article: {entry.link}\n{e}")
    return articles

# Step 3: Collect articles from all sources
records = []
for source, url in feeds.items():
    print(f"Fetching from {source}...")
    for art in fetch_articles(url, max_articles=10):
        records.append({
            'source': source,
            'title': art['title'],
            'text': art['text'],
            'published': art['published']
        })

df = pd.DataFrame(records)
print(f"\nCollected {len(df)} articles total.")

# Step 4: VADER Sentiment Analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['text'].apply(lambda txt: sia.polarity_scores(txt)['compound'])

# Step 5: Visualize results
plt.figure(figsize=(10, 6))
df.boxplot(column='sentiment', by='source')
plt.title('Sentiment Distribution by News Source')
plt.suptitle('')
plt.xlabel('News Source')
plt.ylabel('Compound Sentiment Score')
plt.grid(True)
plt.tight_layout()
plt.show()

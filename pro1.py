"""
# Bias Analysis of Political News on CNN, Fox News, and The New York Times

This script demonstrates how to:
1. Fetch the latest political news articles via RSS feeds.
2. Extract full article text using `newspaper3k`.
3. Perform sentiment analysis with NLTK's VADER as a proxy for bias.
4. Compare sentiment distributions across sources using ANOVA.
5. Visualize results with boxplots.

## Getting Started

1. **Install required packages**:
   ```bash
   pip install feedparser newspaper3k nltk pandas matplotlib scipy
   ```
2. **Download NLTK data**:
   ```bash
   python -m nltk.downloader vader_lexicon
   ```
3. Run this script in a Jupyter Notebook or as a standalone `.py` file.

Feel free to increase the number of articles per source, add more bias metrics (e.g., keyword frequency), or use transformer-based classifiers for deeper analysis.
"""

import feedparser
from newspaper import Article
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Step 1: Define RSS feed URLs for political sections
feeds = {
    'CNN': 'http://rss.cnn.com/rss/cnn_allpolitics.rss',
    'Fox News': 'http://feeds.foxnews.com/foxnews/politics',
    'NYT': 'https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml'
}

# Step 2: Function to fetch and parse articles

def fetch_articles(feed_url, max_articles=20):
    """
    Fetch up to `max_articles` articles from the given RSS feed URL.
    Returns a list of dicts with keys: title, text, published.
    """
    feed = feedparser.parse(feed_url)
    entries = feed.entries[:max_articles]
    articles = []
    for entry in entries:
        try:
            art = Article(entry.link)
            art.download()
            art.parse()
            articles.append({
                'title': entry.title,
                'text': art.text,
                'published': entry.published
            })
        except Exception as e:
            print(f"Error fetching {entry.link}: {e}")
    return articles

# Collect data for all sources
records = []
for source, url in feeds.items():
    print(f"Fetching articles from {source}...")
    for art in fetch_articles(url, max_articles=20):
        records.append({
            'source': source,
            'title': art['title'],
            'text': art['text'],
            'published': art['published']
        })

df = pd.DataFrame(records)
print(f"Collected {len(df)} articles total\n")

# Step 3: Sentiment Analysis using VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['text'].apply(lambda txt: sia.polarity_scores(txt)['compound'])

# Step 4: Visualize sentiment distributions
plt.figure(figsize=(8, 6))
df.boxplot(column='sentiment', by='source')
plt.title('Sentiment Distribution by News Source')
plt.suptitle('')  # remove the automatic subtitle
plt.xlabel('Source')
plt.ylabel('Compound Sentiment Score')
plt.show()

# Step 5: Statistical test (one-way ANOVA)
groups = [group['sentiment'].values for _, group in df.groupby('source')]
stat, p_value = f_oneway(*groups)
print(f"ANOVA results: F = {stat:.2f}, p = {p_value:.3f}")
if p_value < 0.05:
    print("→ There is a statistically significant difference in sentiment across sources.")
else:
    print("→ No significant difference in sentiment detected.")

# ----- Next Steps & Extensions -----
# • Increase `max_articles` for more robust findings.
# • Add keyword-frequency analysis for political terms (e.g., 'Democrat', 'Republican').
# • Incorporate transformer-based bias classifiers (Hugging Face).
# • Compare against real-world benchmarks or implement KL-divergence metrics.

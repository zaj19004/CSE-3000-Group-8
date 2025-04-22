"""
# Bias Analysis via NewsAPI for CNN and Fox News

This script demonstrates how to:
1. Fetch political news from CNN and Fox News using NewsAPI.org.
2. Require an API key stored in an environment variable.
3. Perform sentiment analysis with NLTK's VADER as a proxy for bias.
4. Visualize and statistically compare sentiment distributions across sources.

## Setup and API Key

1. **Install required packages**:
   ```bash
   pip install requests pandas nltk matplotlib scipy
   ```
2. **API Registration**:
   - **NewsAPI.org**: Sign up at https://newsapi.org to get your `NEWSAPI_KEY`.
3. **Environment Variable** (set this in your Codespace):
   ```bash
   export NEWSAPI_KEY="3c5ab6485e724c78a08c3a5de9248cb1"
   ```
4. **Download NLTK data**:
   ```bash
   python -m nltk.downloader vader_lexicon
   ```

## Script
"""
import os
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Load API key from environment
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
if not NEWSAPI_KEY:
    raise RuntimeError('Please set the NEWSAPI_KEY environment variable')

# Step 1: Fetch from NewsAPI for CNN and Fox News

def fetch_from_newsapi(sources, page_size=20):
    """
    Fetch top headlines from specified sources via NewsAPI.
    Returns a list of dicts with keys: source, title, text, published.
    """
    url = 'https://newsapi.org/v2/top-headlines'
    params = {
        'sources': ','.join(sources),
        'pageSize': page_size,
        'apiKey': NEWSAPI_KEY
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    articles = []
    for art in data.get('articles', []):
        # Use content if available, otherwise description
        text = art.get('content') or art.get('description') or ''
        articles.append({
            'source': art['source']['name'],
            'title': art['title'],
            'text': text,
            'published': art['publishedAt']
        })
    return articles

# Collect articles for CNN and Fox News
df_records = []
for rec in fetch_from_newsapi(['cnn', 'fox-news'], page_size=20):
    df_records.append(rec)

# Build DataFrame
df = pd.DataFrame(df_records)
print(f"Collected {len(df)} articles total")

# Step 2: Sentiment Analysis using VADER
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['text'].apply(lambda txt: sia.polarity_scores(txt).get('compound', 0.0))

# Step 3: Visualize sentiment distributions
plt.figure(figsize=(8, 6))
df.boxplot(column='sentiment', by='source')
plt.title('Sentiment Distribution by News Source')
plt.suptitle('')  # remove automatic subtitle
plt.xlabel('Source')
plt.ylabel('Compound Sentiment Score')
plt.savefig('sentiment_boxplot.png', dpi=300, bbox_inches='tight')
print("Saved boxplot to sentiment_boxplot.png")

# Step 4: Statistical test (one-way ANOVA)
groups = [grp['sentiment'].values for _, grp in df.groupby('source')]
stat, p_value = f_oneway(*groups)
print(f"ANOVA results: F = {stat:.2f}, p = {p_value:.3f}")
if p_value < 0.05:
    print("→ Statistically significant difference in sentiment across sources.")
else:
    print("→ No significant difference in sentiment detected.")

# ----- Extensions -----
# • Increase page_size or implement pagination for deeper sampling.
# • Add keyword-frequency analysis for political terms.
# • Incorporate transformer-based political-bias classifiers.

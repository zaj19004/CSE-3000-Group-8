import requests
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize sentiment pipeline
classifier = pipeline("sentiment-analysis")

# Define news platforms and their front pages
news_sites = {
    "CNN": "https://edition.cnn.com/politics",
    "Fox News": "https://www.foxnews.com/politics",
    "NYT": "https://www.nytimes.com/section/politics"
}

def extract_articles(url, max_articles=10):
    """
    Scrapes political article links and texts.
    """
    print(f"Scraping {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    links = list(set(a['href'] for a in soup.find_all('a', href=True) if '/politics/' in a['href']))
    links = [link if link.startswith('http') else f'https://{url.split("//")[1].split("/")[0]}{link}' for link in links[:max_articles]]

    articles = []
    for link in links:
        try:
            article = Article(link)
            article.download()
            article.parse()
            articles.append(article.text[:1000])  # First 1000 chars
        except Exception as e:
            print(f"Skipping article: {e}")
    return articles

# Collect and classify articles
data = []
for outlet, url in news_sites.items():
    texts = extract_articles(url)
    for text in texts:
        try:
            result = classifier(text[:512])[0]
            data.append({
                "outlet": outlet,
                "text": text,
                "label": result["label"],
                "score": result["score"]
            })
        except Exception as e:
            print(f"Error classifying text: {e}")

# Convert to DataFrame
df = pd.DataFrame(data)

# Visualization
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="outlet", hue="label", palette="Set2")
plt.title("Political Sentiment by News Outlet")
plt.xlabel("News Platform")
plt.ylabel("Number of Articles")
plt.legend(title="Sentiment")
plt.tight_layout()
plt.show()
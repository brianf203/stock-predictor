import yfinance as yf
from textblob import TextBlob
from datetime import datetime
import re

class NewsAnalyzer:
    def __init__(self):
        self.bullish_keywords = [
            'surge', 'rally', 'jump', 'soar', 'gain', 'rise', 'up', 'beat', 'exceed',
            'growth', 'profit', 'earnings', 'positive', 'strong', 'bullish', 'buy',
            'upgrade', 'outperform', 'outperform', 'target', 'raise', 'increase',
            'breakthrough', 'innovation', 'expansion', 'acquisition', 'merger',
            'partnership', 'deal', 'contract', 'approval', 'launch', 'record'
        ]
        
        self.bearish_keywords = [
            'drop', 'fall', 'decline', 'plunge', 'crash', 'loss', 'down', 'miss',
            'disappoint', 'negative', 'weak', 'bearish', 'sell', 'downgrade',
            'underperform', 'cut', 'reduce', 'decrease', 'warning', 'concern',
            'risk', 'lawsuit', 'investigation', 'scandal', 'breach', 'hack',
            'recall', 'delay', 'cancel', 'reject', 'deny', 'fail', 'bankruptcy'
        ]
    
    def get_news(self, ticker: str, max_articles: int = 10):
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return []
            
            articles = []
            for item in news[:max_articles]:
                content = item.get('content', {})
                if not content:
                    content = item
                
                title = content.get('title', '') or item.get('title', '') or 'No title available'
                
                canonical_url = content.get('canonicalUrl', {}) or item.get('canonicalUrl', {})
                if isinstance(canonical_url, dict):
                    link = canonical_url.get('url', '')
                else:
                    link = canonical_url or content.get('clickThroughUrl', {}).get('url', '') if isinstance(content.get('clickThroughUrl'), dict) else ''
                
                if not link:
                    click_through = content.get('clickThroughUrl', {}) or item.get('clickThroughUrl', {})
                    if isinstance(click_through, dict):
                        link = click_through.get('url', '')
                
                provider = content.get('provider', {}) or item.get('provider', {})
                if isinstance(provider, dict):
                    publisher = provider.get('displayName', '') or provider.get('name', '')
                else:
                    publisher = str(provider) if provider else ''
                
                if not publisher:
                    publisher = item.get('publisher', '') or item.get('source', '') or 'Yahoo Finance'
                
                pub_date_str = content.get('pubDate', '') or content.get('displayTime', '') or item.get('pubDate', '')
                pub_date = None
                if pub_date_str:
                    try:
                        from dateutil import parser
                        pub_date = parser.parse(pub_date_str)
                    except:
                        try:
                            pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                        except:
                            pass
                
                if not title or title == 'No title available':
                    continue
                
                if not link:
                    link = f"https://finance.yahoo.com/quote/{ticker}/news"
                
                sentiment, sentiment_score = self.analyze_sentiment(title)
                
                articles.append({
                    'title': title,
                    'link': link,
                    'publisher': publisher,
                    'date': pub_date,
                    'sentiment': sentiment,
                    'sentiment_score': sentiment_score
                })
            
            return articles
        except Exception as e:
            print(f"Error fetching news: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def analyze_sentiment(self, text: str):
        if not text:
            return 'neutral', 0.0
        
        text_lower = text.lower()
        
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
        except:
            polarity = 0.0
        
        keyword_score = (bullish_count - bearish_count) / max(len(text.split()), 1)
        combined_score = (polarity * 0.6) + (keyword_score * 0.4)
        
        if combined_score > 0.15:
            sentiment = 'bullish'
        elif combined_score < -0.15:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return sentiment, combined_score
    
    def get_top_news(self, ticker: str, count: int = 3, sentiment_filter: str = None):
        articles = self.get_news(ticker, max_articles=20)
        
        if sentiment_filter:
            articles = [a for a in articles if a['sentiment'] == sentiment_filter]
        
        articles_sorted = sorted(articles, key=lambda x: abs(x['sentiment_score']), reverse=True)
        
        return articles_sorted[:count]
    
    def format_news_for_embed(self, articles):
        if not articles:
            return "No recent news available."
        
        formatted = []
        for i, article in enumerate(articles, 1):
            sentiment_emoji = {
                'bullish': '🟢',
                'bearish': '🔴',
                'neutral': '⚪'
            }.get(article['sentiment'], '⚪')
            
            date_str = article['date'].strftime('%Y-%m-%d') if article['date'] else 'Recent'
            
            formatted.append(
                f"{sentiment_emoji} **{article['title']}**\n"
                f"📅 {date_str} | 📰 {article['publisher']}\n"
                f"🔗 [Read more]({article['link']})"
            )
        
        return '\n\n'.join(formatted)


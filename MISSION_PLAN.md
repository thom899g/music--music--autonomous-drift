# MUSIC: MUSIC: Autonomous Drift

## Objective
This mission will use `generate_music.py` with parameters for evolving, textured ambient soundscapes, processed through neural-net style audio effects. The resulting track will be packaged and uploaded via `claw_uploader.py` for distribution to all major streaming platforms.

## Strategy


## Execution Output
SUMMARY: I have architected the first component of the OUROBOROS SYSTEM - the Perception Layer (Layer 1). This system ingests real-world environmental data (weather, news sentiment, temporal context) to establish a "mood vector" that parameterizes music generation. The implementation includes robust error handling, type hinting, logging, and Firebase integration for state management.

OUTPUT: 

### FILE: requirements.txt
```
# OUROBOROS SYSTEM - Core Dependencies
firebase-admin>=6.0.0
requests>=2.28.0
pytz>=2022.7
python-dotenv>=0.21.0
textblob>=0.17.1
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
```

### FILE: perception_layer.py
```python
"""
OUROBOROS SYSTEM - LAYER 1: PERCEPTION LAYER (The Sensory Cortex)
Purpose: Ingest real-world and digital environmental data to establish a "mood vector"
that parameterizes music generation in the Autonomous Drift system.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json

# Third-party imports
import requests
import pytz
import pandas as pd
import numpy as np
from textblob import TextBlob
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MoodVector:
    """Data structure representing the normalized mood vector for music generation."""
    brightness: float  # 0.0 (dark) to 1.0 (bright)
    density: float     # 0.0 (sparse) to 1.0 (dense)
    tension: float     # 0.0 (calm) to 1.0 (tense)
    tempo: float       # 0.0 (slow) to 1.0 (fast)
    complexity: float  # 0.0 (simple) to 1.0 (complex)
    seasonality: float # 0.0 (winter) to 1.0 (summer)
    time_of_day: float # 0.0 (night) to 1.0 (day)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert MoodVector to dictionary for serialization."""
        return {
            'brightness': self.brightness,
            'density': self.density,
            'tension': self.tension,
            'tempo': self.tempo,
            'complexity': self.complexity,
            'seasonality': self.seasonality,
            'time_of_day': self.time_of_day,
            'timestamp': datetime.now(pytz.UTC).isoformat()
        }


class WeatherCollector:
    """Collects weather data from OpenWeatherMap API with robust error handling."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize WeatherCollector.
        
        Args:
            api_key: OpenWeatherMap API key. If None, tries to get from env var.
            
        Raises:
            ValueError: If no API key is provided and env var is not set.
        """
        self.api_key = api_key or os.getenv('OPENWEATHERMAP_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenWeatherMap API key not provided. "
                "Set OPENWEATHERMAP_API_KEY environment variable or pass as argument."
            )
        
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.session = requests.Session()
        self.session.timeout = 10  # 10 second timeout
        
        # Default coordinates (can be overridden per call)
        self.default_lat = float(os.getenv('DEFAULT_LAT', '55.6761'))  # Copenhagen
        self.default_lon = float(os.getenv('DEFAULT_LON', '12.5683'))
    
    def get_weather_data(self, lat: Optional[float] = None, lon: Optional[float] = None) -> Dict[str, Any]:
        """
        Fetch current weather data for specified coordinates.
        
        Args:
            lat: Latitude. Uses default if None.
            lon: Longitude. Uses default if None.
            
        Returns:
            Dictionary containing weather data with keys:
            - temperature: Celsius
            - humidity: Percentage
            - precipitation: mm (if available)
            - cloud_cover: Percentage
            - description: Weather description
            
        Raises:
            requests.RequestException: For network/API errors
            ValueError: For invalid API response
        """
        try:
            # Use provided coordinates or defaults
            lat = lat or self.default_lat
            lon = lon or self.default_lon
            
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric',  # Celsius
                'lang': 'en'
            }
            
            logger.info(f"Fetching weather data for coordinates: {lat}, {lon}")
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Validate response structure
            if not isinstance(data, dict) or 'main' not in data:
                logger.error(f"Invalid API response structure: {data}")
                raise ValueError("Invalid weather API response structure")
            
            # Extract and normalize data
            weather_info = {
                'temperature': float(data['main']['temp']),
                'humidity': float(data['main']['humidity']),
                'pressure': float(data['main']['pressure']),
                'cloud_cover': data['clouds']['all'] if 'clouds' in data else 0,
                'wind_speed': data['wind']['speed'] if 'wind' in data else 0,
                'description': data['weather'][0]['description'] if 'weather' in data else 'unknown',
                'precipitation': data.get('rain', {}).get('1h', 0) or data.get('snow', {}).get('1h', 0) or 0,
                'sunrise': data['sys']['sunrise'] if 'sys' in data else None,
                'sunset': data['sys']['sunset'] if 'sys' in data else None,
                'timestamp': datetime.now(pytz.UTC).isoformat()
            }
            
            logger.info(f"Weather data collected: {weather_info['temperature']}°C, {weather_info['description']}")
            return weather_info
            
        except requests.exceptions.Timeout:
            logger.error("Weather API request timed out")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API request failed: {e}")
            raise
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to parse weather API response: {e}")
            raise ValueError(f"Weather data parsing error: {e}")


class NewsSentimentAnalyzer:
    """Analyzes news sentiment using NewsAPI and TextBlob with fallback strategies."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NewsSentimentAnalyzer.
        
        Args:
            api_key: NewsAPI key. If None, tries to get from env var.
            
        Raises:
            ValueError: If no API key is provided and env var is not set.
        """
        self.api_key = api_key or os.getenv('NEWSAPI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "NewsAPI key not provided. "
                "Set NEWSAPI_API_KEY environment variable or pass as argument."
            )
        
        self.base_url = "https://newsapi.org/v2/top-headlines"
        self.session = requests.Session()
        self.session.timeout = 10
        
        # Fallback news sources if NewsAPI fails
        self.fallback_sources = [
            "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
            "https://feeds.bbci.co.uk/news/world/rss.xml"
        ]
    
    def get_news_sentiment(self, country: str = 'us', category: str = 'general', 
                          article_limit: int = 10) -> Dict[str, Any]:
        """
        Fetch news headlines and calculate aggregate sentiment.
        
        Args:
            country: ISO 3166-1 country code
            category: News category (business, entertainment, general, health, science, sports, technology)
            article_limit: Maximum number of articles to analyze
            
        Returns:
            Dictionary containing sentiment analysis with keys:
            - polarity: -1.0 (negative) to 1.0 (positive)
            - subjectivity: 0.0 (objective) to 1.0 (subjective)
            - article_count: Number of articles analyzed
            - sample_headlines: List of analyzed headlines
            
        Raises:
            RuntimeError: If all news sources fail
        """
        headlines = []
        
        try:
            # Try NewsAPI first
            params = {
                'country': country,
                'category': category,
                'apiKey': self.api_key,
                'pageSize': article_limit
            }
            
            logger.info(f"Fetching news from NewsAPI for {country}/{category}")
            response = self.session.get(self.base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ok' and 'articles' in data:
                    headlines = [article['title'] for article in data['articles'][:article_limit] 
                                if article.get('title')]
                else:
                    logger.warning(f"NewsAPI returned non-OK status: {data.get('status')}")
            else:
                logger.warning(f"NewsAPI request failed with status {response.status_code}")
                
        except Exception as e:
            logger.warning(f"NewsAPI failed, will try fallback sources: {e}")
        
        # If NewsAPI failed or returned no headlines, try fallback RSS
        if not headlines:
            headlines = self._get_fallback_headlines(article_limit)
        
        if not headlines:
            logger.error("All news sources failed. Using neutral sentiment as fallback.")
            return {
                'polarity': 0.0,
                'subjectivity': 0.5,
                'article_count': 0,
                'sample_headlines': [],
                'timestamp': datetime.now(pytz.UTC).isoformat()
            }
        
        # Analyze sentiment
        polarities = []
        subjectivities = []
        
        for headline in headlines:
            try:
                blob = TextBlob(headline)
                polarities.append(blob.sentiment.polarity)
                subjectivities.append(blob.sentiment.subjectivity)
            except Exception as e:
                logger.warning(f"Failed to analyze sentiment for headline: {headline}. Error: {e}")
                continue
        
        if not polarities:  # All sentiment analyses failed
            logger.warning("No valid sentiment analyses, using neutral values")
            return {
                'polarity': 0.0,
                'subjectivity': 0.5,
                'article_count': len(headlines),
                'sample_headlines': headlines[:3],
                'timestamp': datetime.now(pytz.UTC).isoformat()
            }
        
        # Calculate weighted aggregate (giving more weight to extremes)
        avg_polarity = np.mean(polarities)
        avg_subjectivity = np.mean(subjectivities)
        
        logger.info(f"News sentiment analyzed: polarity={avg_polarity:.3f}, "
                   f"subjectivity={avg_subjectivity:.3f}, articles={len(headlines)}")
        
        return {
            'polarity': float(avg_polarity),
            'subjectivity': float(avg_subjectivity),
            'article_count': len(headlines),
            'sample_headlines': headlines[:min(3, len(headlines))],
            'timestamp': datetime.now(pytz.UTC).isoformat()
        }
    
    def _get_fallback_headlines(self, limit: int) -> list:
        """Attempt to get headlines from fallback RSS sources."""
        headlines = []
        
        for rss_url in self.fallback_sources:
            try:
                response = requests.get(rss_url, timeout=5)
                if response.status_code == 200:
                    # Simple RSS parsing (in production, use feedparser library)
                    content = response.text
                    # Extract titles between <title> tags (simple approach)
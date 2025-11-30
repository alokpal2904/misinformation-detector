import requests
import os
from datetime import datetime
from datetime import timedelta
import time
from dotenv import load_dotenv
import json

load_dotenv()

class RedditScraper:
    def __init__(self):
        self.trending_topics = []
        self.last_check = datetime.now()
        self.access_token = None
        self.token_expires = datetime.now()
        
        # Try to authenticate if credentials are available
        if os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET"):
            self.authenticate()
    
    def authenticate(self):
        try:
            auth = requests.auth.HTTPBasicAuth(
                os.getenv("REDDIT_CLIENT_ID"), 
                os.getenv("REDDIT_CLIENT_SECRET")
            )
            
            data = {
                'grant_type': 'client_credentials'
            }
            
            headers = {
                'User-Agent': os.getenv("REDDIT_USER_AGENT", "python:misinformation-detector:v1.0")
            }
            
            response = requests.post(
                'https://www.reddit.com/api/v1/access_token',
                auth=auth,
                data=data,
                headers=headers
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                # Set expiration time (usually 1 hour)
                self.token_expires = datetime.now() + time.timedelta(seconds=token_data['expires_in'] - 60)
                print("Successfully authenticated with Reddit API")
            else:
                print(f"Failed to authenticate with Reddit API: {response.status_code}")
        except Exception as e:
            print(f"Error authenticating with Reddit: {e}")
    
    def get_trending_posts(self, subreddit_name, limit=10):
        try:
            # Check if we need to reauthenticate
            if self.access_token is None or datetime.now() >= self.token_expires:
                self.authenticate()
            
            if self.access_token is None:
                print("No valid authentication for Reddit API")
                return []
            
            headers = {
                'Authorization': f'bearer {self.access_token}',
                'User-Agent': os.getenv("REDDIT_USER_AGENT", "python:misinformation-detector:v1.0")
            }
            
            response = requests.get(
                f'https://oauth.reddit.com/r/{subreddit_name}/hot',
                headers=headers,
                params={'limit': limit}
            )
            
            if response.status_code == 200:
                posts = response.json()['data']['children']
                trending_posts = []
                
                for post in posts:
                    post_data = post['data']
                    # Consider posts with high engagement
                    if post_data['score'] > 100 and post_data['num_comments'] > 50:
                        trending_posts.append({
                            'title': post_data['title'],
                            'url': post_data['url'],
                            'score': post_data['score'],
                            'num_comments': post_data['num_comments'],
                            'created_utc': post_data['created_utc'],
                            'id': post_data['id'],
                            'subreddit': subreddit_name
                        })
                
                return trending_posts
            else:
                print(f"Error getting posts from {subreddit_name}: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error getting posts from {subreddit_name}: {e}")
            return []
    
    def run(self):
        while True:
            try:
                # Check popular subreddits
                subreddits = ['news', 'worldnews', 'politics', 'technology', 'science']
                current_trending = []
                
                for sub in subreddits:
                    posts = self.get_trending_posts(sub)
                    current_trending.extend(posts)
                
                # Update trending topics
                self.trending_topics = current_trending
                self.last_check = datetime.now()
                
                print(f"Found {len(current_trending)} trending posts at {self.last_check}")
                
                # Wait for 2 minutes
                time.sleep(120)
            except Exception as e:
                print(f"Error in RedditScraper: {e}")
                time.sleep(60)
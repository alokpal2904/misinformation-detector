import threading
import queue
import json
import time
from datetime import datetime
from reddit_scraper import RedditScraper
from vector_db import VectorDatabase
from fact_checker import FactCheckerAgent

class MisinformationDetector:
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.reddit_scraper = RedditScraper()
        self.fact_checker = FactCheckerAgent(self.vector_db)
        self.claims_queue = queue.Queue()
        self.verified_claims = []
        
        # Load known misinformation at startup
        self.load_known_misinformation()
    
    def load_known_misinformation(self):
        # Load from file or database in real implementation
        known_claims = [
            {"text": "COVID-19 vaccines contain microchips", "is_misinformation": True, "sources": ["WHO", "CDC"]},
            {"text": "The moon landing was faked", "is_misinformation": True, "sources": ["NASA", "scientific consensus"]},
            {"text": "Climate change is a hoax", "is_misinformation": True, "sources": ["IPCC", "NASA climate data"]},
            # Add more known claims
        ]
        self.vector_db.add_known_misinformation(known_claims)
    
    def start_reddit_monitor(self):
        def monitor():
            while True:
                trending = self.reddit_scraper.trending_topics
                for post in trending:
                    # Check if we've already processed this
                    if not any(v.get('post', {}).get('id') == post['id'] for v in self.verified_claims):
                        print(f"Adding claim to queue: {post['title']}")
                        self.claims_queue.put(post)
                time.sleep(30)  # Check every 30 seconds
        
        thread = threading.Thread(target=monitor)
        thread.daemon = True
        thread.start()
    
    def start_fact_checking(self):
        def process_queue():
            while True:
                try:
                    claim = self.claims_queue.get(timeout=10)
                    print(f"Processing claim: {claim['title']}")
                    
                    result = self.fact_checker.verify_claim(claim['title'])
                    
                    # Add post information to the result
                    result['post'] = claim
                    result['timestamp'] = datetime.now().isoformat()
                    
                    self.verified_claims.append(result)
                    
                    print(f"Verified claim: {claim['title']} - Verdict: {result['verdict']}")
                    
                    # Add to vector database for future reference
                    if result['verdict'] in ['false', 'misleading']:
                        self.vector_db.add_known_misinformation([{
                            'text': claim['title'],
                            'is_misinformation': True,
                            'sources': result.get('sources', [])
                        }])
                
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error in fact-checking: {e}")
        
        thread = threading.Thread(target=process_queue)
        thread.daemon = True
        thread.start()
    
    def run(self):
        # Start Reddit scraper in background
        reddit_thread = threading.Thread(target=self.reddit_scraper.run)
        reddit_thread.daemon = True
        reddit_thread.start()
        
        # Wait a bit for the scraper to initialize
        time.sleep(5)
        
        # Start monitoring for trending topics
        self.start_reddit_monitor()
        
        # Start fact-checking
        self.start_fact_checking()
        
        print("Misinformation detector started successfully!")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
    
    def get_recent_claims(self, limit=10):
        """Get recent verified claims for the dashboard"""
        return self.verified_claims[-limit:] if self.verified_claims else []



# agents/misinformation_detector.py
import os
import asyncio
import json
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
import praw
from bs4 import BeautifulSoup
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

load_dotenv()

# Initialize models
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="meta-llama/llama-3-70b-instruct:nitro"
)

perplexity_llm = ChatOpenAI(
    base_url="https://api.perplexity.ai",
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    model="llama-3-sonar-large-32k-online"
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize FAISS index
dimension = 768  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# Define state
class AgentState:
    def __init__(self):
        self.messages = []
        self.reddit_posts = []
        self.verified_posts = []
        self.current_post = None
        self.verification_results = {}
        self.search_results = {}

# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web for current information using SerpAPI."""
    search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
    return search.run(query)

@tool
def verify_with_perplexity(claim: str) -> str:
    """Verify a claim using Perplexity AI's online search capabilities."""
    prompt = f"""
    Verify the following claim with current information from reliable sources: 
    "{claim}"
    
    Provide a detailed analysis with sources. Determine if the claim is:
    - True
    - Mostly True
    - Mixed Evidence
    - Mostly False
    - False
    - Unverifiable
    
    Include your reasoning and sources.
    """
    
    response = perplexity_llm.invoke([HumanMessage(content=prompt)])
    return response.content

@tool
def scrape_reddit_trending(subreddit: str = "all", limit: int = 10) -> List[Dict]:
    """Scrape trending posts from Reddit."""
    try:
        subreddit_obj = reddit.subreddit(subreddit)
        trending_posts = []
        
        for post in subreddit_obj.hot(limit=limit):
            # Skip stickied posts (often rules or announcements)
            if post.stickied:
                continue
                
            trending_posts.append({
                "id": post.id,
                "title": post.title,
                "url": post.url,
                "score": post.score,
                "num_comments": post.num_comments,
                "created_utc": post.created_utc,
                "author": str(post.author),
                "subreddit": str(post.subreddit),
                "selftext": post.selftext,
                "permalink": f"https://reddit.com{post.permalink}"
            })
        
        return trending_posts
    except Exception as e:
        print(f"Error scraping Reddit: {e}")
        return []

@tool
def analyze_claim_context(claim: str) -> Dict[str, Any]:
    """Analyze the contextual meaning of a claim to understand its nuances."""
    prompt = f"""
    Analyze the following claim for contextual understanding. Don't just extract keywords.
    Identify the main entities, relationships, implied meanings, and potential for misinformation.
    
    Claim: "{claim}"
    
    Return your analysis as a JSON object with:
    - main_entities: list of main entities mentioned
    - relationships: description of how entities are related
    - implied_meanings: any implied or hidden meanings
    - potential_misinformation_indicators: reasons why this might be misinformation
    - contextual_understanding: a comprehensive summary of the claim's meaning in context
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        # Try to parse JSON from response
        return json.loads(response.content)
    except:
        # If not JSON, return as text
        return {"analysis": response.content}

# Define agent nodes
def reddit_scraper_agent(state: AgentState) -> Dict:
    """Agent that scrapes trending Reddit posts."""
    print("Scraping Reddit for trending posts...")
    posts = scrape_reddit_trending("all", 10)
    
    # Filter to get the most engaging posts
    engaging_posts = sorted(posts, key=lambda x: x['score'] + x['num_comments'], reverse=True)[:5]
    
    return {"reddit_posts": engaging_posts}

def content_analyzer_agent(state: AgentState) -> Dict:
    """Agent that analyzes Reddit posts for potential misinformation."""
    print("Analyzing Reddit posts for potential misinformation...")
    
    analyzed_posts = []
    for post in state.reddit_posts:
        # Combine title and text for analysis
        content = f"{post['title']}. {post.get('selftext', '')}"
        
        # Analyze context
        context_analysis = analyze_claim_context(content)
        
        # Check if this is similar to known misinformation
        is_potential_misinfo = check_misinformation_similarity(content)
        
        analyzed_posts.append({
            **post,
            "context_analysis": context_analysis,
            "potential_misinformation": is_potential_misinfo
        })
    
    # Sort by potential misinformation score
    prioritized_posts = sorted(
        analyzed_posts, 
        key=lambda x: x["potential_misinformation"]["similarity_score"], 
        reverse=True
    )[:5]  # Top 5 most likely misinformation
    
    return {"analyzed_posts": prioritized_posts}

def check_misinformation_similarity(content: str) -> Dict[str, Any]:
    """Check if content is similar to known misinformation using embeddings."""
    # Generate embedding for the content
    content_embedding = embeddings.embed_query(content)
    content_embedding = np.array([content_embedding]).astype('float32')
    
    # Search in FAISS index (in a real system, this would be populated with known misinformation)
    if index.ntotal > 0:
        distances, indices = index.search(content_embedding, 3)
        similarity_score = float(1 - distances[0][0]) if distances[0][0] > 0 else 0
    else:
        # If no known misinformation in index, use a heuristic based on content characteristics
        similarity_score = heuristic_misinfo_score(content)
    
    return {
        "similarity_score": similarity_score,
        "requires_verification": similarity_score > 0.3
    }

def heuristic_misinfo_score(content: str) -> float:
    """Heuristic score for misinformation potential based on content characteristics."""
    indicators = [
        ("conspiracy", 0.7),
        ("breaking", 0.6),
        ("shocking", 0.6),
        ("you won't believe", 0.8),
        ("government hiding", 0.7),
        ("they don't want you to know", 0.8),
        (" miracle", 0.5),
        ("cure", 0.6)
    ]
    
    content_lower = content.lower()
    score = 0
    
    for indicator, weight in indicators:
        if indicator in content_lower:
            score += weight
    
    # Normalize to 0-1 range
    return min(score / 5, 1.0)

def fact_checking_agent(state: AgentState) -> Dict:
    """Agent that coordinates fact-checking of suspicious claims."""
    print("Initiating fact-checking process...")
    
    verification_results = {}
    for post in state.analyzed_posts:
        if post["potential_misinformation"]["requires_verification"]:
            claim = f"{post['title']}. {post.get('selftext', '')}"
            post_id = post["id"]
            
            print(f"Verifying claim: {claim[:100]}...")
            
            # Search web for information
            search_query = f"fact check {post['title']}"
            search_results = search_web(search_query)
            
            # Verify with Perplexity
            verification = verify_with_perplexity(claim)
            
            verification_results[post_id] = {
                "claim": claim,
                "search_results": search_results,
                "verification": verification,
                "verification_time": datetime.now().isoformat()
            }
    
    return {"verification_results": verification_results}

def report_generator_agent(state: AgentState) -> Dict:
    """Agent that generates reports from verification results."""
    print("Generating verification reports...")
    
    verified_posts = []
    for post in state.analyzed_posts:
        post_id = post["id"]
        
        if post_id in state.verification_results:
            verification = state.verification_results[post_id]
            
            # Generate a summary of the verification
            summary_prompt = f"""
            Based on the following verification data, create a concise summary:
            
            Claim: {verification['claim']}
            Search Results: {verification['search_results']}
            Perplexity Verification: {verification['verification']}
            
            Create a structured report with:
            1. Claim summary
            2. Verification status (True, Mostly True, Mixed, Mostly False, False, Unverifiable)
            3. Key evidence
            4. Confidence level
            5. Recommended action
            
            Format the response as JSON.
            """
            
            response = llm.invoke([HumanMessage(content=summary_prompt)])
            
            try:
                report = json.loads(response.content)
            except:
                report = {"summary": response.content}
            
            verified_posts.append({
                **post,
                "verification_report": report,
                "verification_data": verification
            })
    
    return {"verified_posts": verified_posts}

# Build the graph
def create_misinformation_detection_workflow():
    """Create the LangGraph workflow for misinformation detection."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("reddit_scraper", reddit_scraper_agent)
    workflow.add_node("content_analyzer", content_analyzer_agent)
    workflow.add_node("fact_checker", fact_checking_agent)
    workflow.add_node("report_generator", report_generator_agent)
    
    # Add edges
    workflow.set_entry_point("reddit_scraper")
    workflow.add_edge("reddit_scraper", "content_analyzer")
    workflow.add_edge("content_analyzer", "fact_checker")
    workflow.add_edge("fact_checker", "report_generator")
    workflow.add_edge("report_generator", END)
    
    return workflow.compile()

# Initialize the workflow
misinformation_workflow = create_misinformation_detection_workflow()
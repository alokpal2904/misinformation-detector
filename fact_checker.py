from langchain.agents import AgentType, initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
import requests
from bs4 import BeautifulSoup
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

class FactCheckerAgent:
    def __init__(self, vector_db):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0
        )
        self.vector_db = vector_db
        self.tools = self.setup_tools()
        
        # Use a simpler approach without deprecated agent initialization
        self.system_message = SystemMessage(content="""
        You are a fact-checking assistant. Analyze the provided claim and determine its veracity.
        Use available tools to research the claim. Consider:
        1. Credibility of sources
        2. Consistency with established facts
        3. Evidence supporting or refuting the claim
        4. Potential biases or agendas
        
        Return a JSON object with:
        - verdict: "true", "false", "misleading", or "unverifiable"
        - confidence: 0-100
        - explanation: brief explanation of your reasoning
        - sources: list of sources consulted
        """)
    
    def setup_tools(self):
        def web_search_tool(query):
            """Search the web for current information about a topic"""
            try:
                params = {
                    "q": query,
                    "api_key": os.getenv("SERPAPI_API_KEY"),
                    "engine": "google",
                    "num": 5
                }
                response = requests.get("https://serpapi.com/search", params=params, timeout=10)
                results = response.json().get("organic_results", [])
                return str([r.get("snippet", "") for r in results])
            except Exception as e:
                return f"Error in web search: {str(e)}"
        
        def check_known_claims_tool(query):
            """Check if similar claims have been verified before"""
            try:
                results = self.vector_db.search_similar_claims(query)
                return str([f"Claim: {r.page_content}, Metadata: {r.metadata}" for r in results])
            except Exception as e:
                return f"Error checking known claims: {str(e)}"
        
        def extract_content_tool(url):
            """Extract main content from a webpage"""
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                return text[:3000]  # Limit to first 3000 characters
            except Exception as e:
                return f"Could not extract content from URL: {str(e)}"
        
        tools = [
            Tool(
                name="WebSearch",
                func=web_search_tool,
                description="Useful for searching the web for current information about a topic"
            ),
            Tool(
                name="CheckKnownClaims",
                func=check_known_claims_tool,
                description="Useful for checking if similar claims have been verified before"
            ),
            Tool(
                name="ExtractWebContent",
                func=extract_content_tool,
                description="Useful for extracting main content from a webpage URL"
            )
        ]
        
        return tools
    
    def extract_json_from_response(self, text):
        """Try to extract JSON from the agent response"""
        try:
            # Look for JSON pattern in the response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # If no JSON found, create a basic response
                return {
                    "verdict": "unverifiable",
                    "confidence": 0,
                    "explanation": "Could not parse verification result",
                    "sources": []
                }
        except json.JSONDecodeError:
            return {
                "verdict": "unverifiable",
                "confidence": 0,
                "explanation": "Invalid JSON response from agent",
                "sources": []
            }
    
    def verify_claim(self, claim_text):
        # Simple implementation without the deprecated agent
        # In a real implementation, you'd use LangGraph or another approach
        prompt = f"""
        Please fact-check the following claim: "{claim_text}"
        
        Use the available tools to research this claim and provide a JSON response with:
        - verdict: "true", "false", "misleading", or "unverifiable"
        - confidence: 0-100
        - explanation: brief explanation of your reasoning
        - sources: list of sources consulted
        """
        
        try:
            # For now, we'll use a simple approach
            # In a real implementation, you'd use the tools here
            response = self.llm.invoke([
                self.system_message,
                {"role": "user", "content": prompt}
            ])
            
            return self.extract_json_from_response(response.content)
        except Exception as e:
            print(f"Error in fact-checking: {e}")
            return {
                "verdict": "unverifiable",
                "confidence": 0,
                "explanation": f"Error during verification: {str(e)}",
                "sources": []
            }
import os
import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
import json
import wikipedia
import re

load_dotenv()

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    st.warning("Please install the latest HuggingFace integration: `pip install -U langchain-huggingface`")

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

if not openrouter_api_key:
    st.error("Please set OPENROUTER_API_KEY in your .env file")
    st.stop()

os.environ["USER_AGENT"] = "MisinformationDetector/1.0 (abc@gmail.com)"
wikipedia.set_user_agent("MisinformationDetector/1.0 (abc@gmail.com)")

def initialize_components():
    llm = ChatOpenAI(
        model="openai/gpt-3.5-turbo",
        temperature=0,
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )
    
    return llm

def enhanced_wikipedia_search(query):
    """Enhanced Wikipedia search with better error handling"""
    try:
        try:
            summary = wikipedia.summary(query, sentences=5)
            return f"Wikipedia Summary: {summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                summary = wikipedia.summary(e.options[0], sentences=5)
                return f"Wikipedia Summary for '{e.options[0]}': {summary}"
            except:
                return f"Wikipedia Disambiguation: {e.options[:5]}"
        except wikipedia.exceptions.PageError:
            search_results = wikipedia.search(query, results=3)
            return f"Wikipedia Search Results: {', '.join(search_results)}"
    except Exception as e:
        return f"Error with Wikipedia search: {str(e)}"

def perplexity_search(query):
    """Search using Perplexity API for high-quality web results"""
    try:
        if not perplexity_api_key:
            return "Perplexity API not configured. Please set PERPLEXITY_API_KEY in your .env file."
        
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an accurate and factual search assistant. Provide concise, factual information with sources."
                },
                {
                    "role": "user",
                    "content": f"Search for factual information about: {query}. Provide specific details and sources."
                }
            ],
            "max_tokens": 1000
        }
        headers = {
            "Authorization": f"Bearer {perplexity_api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        return result['choices'][0]['message']['content']
        
    except Exception as e:
        return f"Error with Perplexity search: {str(e)}"

def enhanced_duckduckgo_search(query):
    """Enhanced DuckDuckGo search with better formatting"""
    try:
        search = DuckDuckGoSearchAPIWrapper()
        results = search.run(query)
        
        if not results or "No good DuckDuckGo" in results:
            return f"DuckDuckGo Search for '{query}': No direct results. Try rephrasing your query."
        
        return f"DuckDuckGo Results: {results}"
    except Exception as e:
        return f"Error with DuckDuckGo search: {str(e)}"

def create_verification_tools():
    """Create tools for the verification agent"""
    wikipedia_tool = Tool(
        name="WikipediaSearch",
        func=enhanced_wikipedia_search,
        description="Useful for retrieving factual information from Wikipedia about people, places, events, and concepts"
    )
    
    perplexity_tool = Tool(
        name="PerplexityWebSearch",
        func=perplexity_search,
        description="Useful for retrieving high-quality web search results with sources for factual verification"
    )
    
    duckduckgo_tool = Tool(
        name="DuckDuckGoWebSearch",
        func=enhanced_duckduckgo_search,
        description="Useful for retrieving general web search results for fact-checking"
    )
    
    return [wikipedia_tool, perplexity_tool, duckduckgo_tool]

def create_verification_agent(llm, tools):
    """Create a ReAct agent for claim verification"""
   
    prompt_template = """You are an expert fact-checker and misinformation detection agent. 
Your goal is to verify claims by using available tools to gather evidence from multiple sources.

IMPORTANT: You MUST use ALL available tools to research the claim thoroughly:
1. First, use WikipediaSearch to get background information
2. Then, use PerplexityWebSearch for high-quality web results
3. Finally, use DuckDuckGoWebSearch for general web search results

After using all tools, provide your final verification.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat until you've used all tools)
Thought: I have used all tools and now know the final answer
Final Answer: [your final answer in the required format]

Your Final Answer must be in this exact format:
- VERIFICATION STATUS: [True/False/Mixed/Unverified]
- CONFIDENCE LEVEL: [High/Medium/Low]
- EVIDENCE SUMMARY: [brief summary of key evidence]
- DETAILED EXPLANATION: [detailed explanation of your reasoning]

Begin! Remember to use ALL available tools.

Question: {input}
{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(prompt_template)
    
    # Create the ReAct agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    
    # Create agent executor with improved configuration
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,  # Increased to allow for all tools to be used
        return_intermediate_steps=True
    )
    
    return agent_executor

def parse_agent_output(output_text):
    """Parse the agent's output to extract the structured verification"""
    # Try to extract the structured parts
    status_match = re.search(r"VERIFICATION STATUS:\s*(.+)", output_text, re.IGNORECASE)
    confidence_match = re.search(r"CONFIDENCE LEVEL:\s*(.+)", output_text, re.IGNORECASE)
    summary_match = re.search(r"EVIDENCE SUMMARY:\s*(.+)", output_text, re.IGNORECASE)
    explanation_match = re.search(r"DETAILED EXPLANATION:\s*(.+)", output_text, re.IGNORECASE)
    
    # If we can't parse the structured output, return the raw text
    if not all([status_match, confidence_match, summary_match, explanation_match]):
        return output_text
    
    # Format the parsed output
    parsed_output = f"""
- VERIFICATION STATUS: {status_match.group(1).strip()}
- CONFIDENCE LEVEL: {confidence_match.group(1).strip()}
- EVIDENCE SUMMARY: {summary_match.group(1).strip()}
- DETAILED EXPLANATION: {explanation_match.group(1).strip()}
"""
    
    return parsed_output

def process_claim_with_agent(claim):
    """Process a claim using the agentic approach"""
    # Initialize components
    llm = initialize_components()
    
    # Create tools
    tools = create_verification_tools()
    
    # Create agent
    agent = create_verification_agent(llm, tools)
    
    # Create prompt for the agent
    agent_prompt = f"Verify this claim: {claim}"
    
    # Execute the agent
    with st.spinner("🤖 Agent is researching and analyzing the claim..."):
        try:
            result = agent.invoke({"input": agent_prompt})
            
            # Extract the final answer and intermediate steps
            final_answer = result["output"] if "output" in result else "Agent completed but did not provide a clear verification."
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Parse the final answer to ensure proper formatting
            parsed_answer = parse_agent_output(final_answer)
            
            # Extract tool outputs from intermediate steps
            tool_outputs = {}
            for step in intermediate_steps:
                if len(step) >= 2:
                    action = step[0]
                    observation = step[1]
                    if hasattr(action, 'tool'):
                        tool_name = action.tool
                        tool_outputs[tool_name] = observation
            
            # If any tools weren't used, run them now to ensure we have all evidence
            all_tools = {tool.name: tool for tool in tools}
            for tool_name in all_tools:
                if tool_name not in tool_outputs:
                    try:
                        tool_outputs[tool_name] = all_tools[tool_name].func(claim)
                    except Exception as e:
                        tool_outputs[tool_name] = f"Error running {tool_name}: {str(e)}"
            
            return {
                "verification": parsed_answer,
                "tool_outputs": tool_outputs
            }
                
        except Exception as e:
            error_msg = f"Error during agent execution: {str(e)}"
            st.error(error_msg)
            return {
                "verification": f"""
                VERIFICATION STATUS: Error
                CONFIDENCE LEVEL: N/A
                EVIDENCE SUMMARY: Unable to complete verification due to technical error.
                DETAILED EXPLANATION: {error_msg}
                
                Please try again or check your API configurations.
                """,
                "tool_outputs": {}
            }

# Streamlit UI
def main():
    st.title("🔍 AI Agentic Misinformation Detector")
    st.write("Enter a claim to verify its accuracy using AI agents that dynamically research multiple sources")
    
    # Text input for claim
    claim = st.text_area("Enter the claim to verify:", height=100, 
                         placeholder="e.g., 'The moon landing was faked' or 'Vaccines cause autism'")
    
    if st.button("🚀 Verify Claim with AI Agent", type="primary"):
        if claim:
            result = process_claim_with_agent(claim)
            
            st.subheader("📊 Agent Verification Result")
            
            # Display the agent's verification result
            st.write(result["verification"])
            
            # Display tool outputs if available
            if result["tool_outputs"]:
                st.subheader("🔍 Evidence Sources")
                
                # Display Wikipedia results
                if "WikipediaSearch" in result["tool_outputs"]:
                    with st.expander("📚 Wikipedia Results"):
                        st.text_area("Wikipedia Evidence", result["tool_outputs"]["WikipediaSearch"], height=150, key="wikipedia_evidence")
                
                # Display Perplexity results
                if "PerplexityWebSearch" in result["tool_outputs"]:
                    with st.expander("🤖 Perplexity AI Results"):
                        st.text_area("Perplexity Evidence", result["tool_outputs"]["PerplexityWebSearch"], height=150, key="perplexity_evidence")
                
                # Display DuckDuckGo results
                if "DuckDuckGoWebSearch" in result["tool_outputs"]:
                    with st.expander("🌐 DuckDuckGo Results"):
                        st.text_area("DuckDuckGo Evidence", result["tool_outputs"]["DuckDuckGoWebSearch"], height=150, key="duckduckgo_evidence")
            
            # Add download option
            report_data = result["verification"]
            if result["tool_outputs"]:
                report_data += "\n\n=== EVIDENCE SOURCES ===\n"
                for tool_name, output in result["tool_outputs"].items():
                    report_data += f"\n--- {tool_name} ---\n{output}\n"
            
            st.download_button(
                label="📥 Download Verification Report",
                data=report_data,
                file_name=f"verification_report.txt",
                mime="text/plain"
            )
        else:
            st.warning("Please enter a claim to verify.")

if __name__ == "__main__":
    main()
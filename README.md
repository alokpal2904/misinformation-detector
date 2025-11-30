# 🔍 AI Agentic Misinformation Detector

An intelligent claim verification system powered by AI agents that dynamically research multiple authoritative sources to detect misinformation with transparency and accuracy.

## 🌟 Features

- **AI Agentic Verification**: Uses LangChain ReAct agent that autonomously researches claims across multiple sources
- **Multi-Source Verification**: Integrates Wikipedia, Perplexity AI, Google News, and DuckDuckGo for comprehensive fact-checking
- **Negation-Aware Parsing**: Correctly evaluates claims containing negation (not, never, no, etc.) to avoid misinterpretation
- **Structured Output**: Returns verification status, confidence level, evidence summary, and detailed explanations
- **Transparent Evidence**: Displays intermediate tool outputs and sources for complete auditability
- **Downloadable Reports**: Export verification results and evidence sources as text files
- **User-Friendly Interface**: Built with Streamlit for intuitive claim verification

## 🛠️ Tech Stack

- **Python** - Core language
- **Streamlit** - Web interface
- **LangChain** - AI agent orchestration
- **OpenRouter (ChatOpenAI)** - LLM backbone (GPT-3.5-Turbo)
- **Perplexity API** - High-quality web search
- **Wikipedia API** - Factual information retrieval
- **Google News** - Current event verification
- **DuckDuckGo** - General web search
- **HuggingFace** - Embeddings support
- **dotenv** - Environment variable management

## 📋 Prerequisites

- Python 3.8+
- API Keys:
  - OpenRouter API Key (for LLM access)
  - Perplexity API Key (for web search)
  - Google News API (included with gnews library)

## 🚀 Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd "c:\Users\alokp\Documents\Misinformation Detector"
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** - Create a `.env` file:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key
   PERPLEXITY_API_KEY=your_perplexity_api_key
   ```

   Or configure Streamlit secrets at `~/.streamlit/secrets.toml`:
   ```toml
   OPENROUTER_API_KEY = "your_openrouter_api_key"
   PERPLEXITY_API_KEY = "your_perplexity_api_key"
   ```

## 📦 Requirements

```
streamlit
langchain
langchain-community
langchain-openai
langchain-huggingface
wikipedia
duckduckgo-search
gnews
requests
python-dotenv
```

## ▶️ Usage

1. **Run the application**:
   ```bash
   streamlit run main.py
   ```

2. **Enter a claim** in the text area (e.g., "The Earth is flat")

3. **Click "Verify Claim with AI Agent"** to initiate verification

4. **Review Results**:
   - Verification status (True/False/Mixed/Unverified)
   - Confidence level
   - Evidence summary
   - Detailed explanation
   - Individual tool outputs from all sources

5. **Download Report** using the export button to save findings

## 🧠 How It Works

1. **Claim Analysis**: Detects negation words to understand claim intent
2. **Agent Initialization**: Creates a ReAct agent with access to 4 search tools
3. **Multi-Source Research**: Agent autonomously uses all tools to gather evidence
4. **Evidence Evaluation**: LLM synthesizes findings and evaluates claim truthfulness
5. **Result Formatting**: Structures output with status, confidence, and explanations
6. **Transparency**: Displays all intermediate steps and source evidence

## 🔑 Key Components

- **`enhanced_wikipedia_search()`**: Handles Wikipedia queries with disambiguation error handling
- **`perplexity_search()`**: Queries Perplexity AI for high-quality web results
- **`google_news_search()`**: Retrieves relevant news articles for current events
- **`enhanced_duckduckgo_search()`**: General web search with fallback handling
- **`contains_negation()`**: Detects negation words to interpret claims correctly
- **`create_verification_agent()`**: Builds ReAct agent with custom prompt templates
- **`process_claim_with_agent()`**: Orchestrates claim verification workflow

## 🎯 Example Usage

**Claim**: "COVID-19 is not dangerous"

**Process**:
1. Agent detects negation ("not")
2. Researches COVID-19 severity across all sources
3. Evidence shows COVID-19 IS dangerous
4. Status: **False** (negated claim contradicted by evidence)
5. Confidence: **High**

## 📊 Output Format

```
- VERIFICATION STATUS: True/False/Mixed/Unverified
- CONFIDENCE LEVEL: High/Medium/Low
- EVIDENCE SUMMARY: Brief summary of key findings
- DETAILED EXPLANATION: Detailed reasoning and analysis
```

## ⚙️ Configuration

- **Model**: OpenAI GPT-3.5-Turbo via OpenRouter
- **Temperature**: 0 (deterministic responses)
- **Max Agent Iterations**: 12
- **Tool Timeout**: 30 seconds per request
- **User Agent**: MisinformationDetector/1.0

## 🔐 Security

- API keys stored in `.env` or Streamlit secrets (not in code)
- No sensitive data logged or exposed
- All API calls use secure HTTPS connections
- User claims are not stored or shared

## 📝 Limitations

- Verification quality depends on source availability
- Complex claims may require "Mixed" verification status
- Real-time events have a slight delay in reporting
- Perplexity API requires valid subscription

## 🤝 Contributing

Feel free to extend functionality by:
- Adding more verification tools
- Improving negation detection
- Enhancing prompt templates
- Adding support for multi-language claims



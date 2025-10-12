***Unless specified otherwise in the AIIP_exceptions folder of this repository, this project follows the guidelines outlined in the AI-Internship-Program repository.***

---
# 🌟 Strategist-small


## 🏗️ Architecture

**LlamaIndex for RAG + LangChain for CrewAI Integration**

Here's why this combination is optimal:

## 📋 Requirements Breakdown

1.  **Complex document parsing** (PDFs, tables, images, multiple formats)
    -   ✅ **LlamaIndex excels here** - best-in-class document parsing
2.  **Multiple data sources** (different types of documents)
    -   ✅ **LlamaIndex** - handles 100+ data connectors
3.  **CrewAI agents** querying the RAG
    -   ✅ **LangChain** - CrewAI is built on LangChain, native integration
4.  **Complex analysis workflows**
    -   ✅ **CrewAI + LangChain** - designed for multi-agent orchestration


## 🎯 Why This Hybrid Approach?
| Component          | Technology         | Reason                                               |
|--------------------|--------------------|------------------------------------------------------|
| Document Parsing   | LlamaIndex         | Best PDF/table/image parsing, 100+ connectors        |
| RAG Index          | LlamaIndex         | Superior chunking, metadata, query engines           |
| Agent Framework    | CrewAI (LangChain) | Built for multi-agent workflows                      |
| RAG ↔ Agents       | Custom Tool        | Bridges LlamaIndex to CrewAI                         |
| LLM                | Groq               | Fast inference for agents                            |




## 📊 Architecture Benefits

**✅ LlamaIndex for RAG:**

-   Handles complex PDFs with tables/images
-   Multiple data source connectors (web, databases, APIs)
-   Advanced chunking strategies
-   Metadata filtering
-   Query routing and sub-questions

**✅ CrewAI for Analysis:**

-   Multiple specialized agents working together
-   Sequential or hierarchical workflows
-   Each agent can query the RAG independently
-   Built-in task delegation and memory

**✅ Bridge Tool:**

-   Simple `@tool` decorator exposes RAG to agents
-   Agents call `search_knowledge_base("query")`
-   Clean separation of concerns

## 🚀 Workflow

```
1. Parse Documents (LlamaIndex)
   ├── PDFs with tables → LlamaIndex PDF Reader
   ├── Word docs → LlamaIndex DOCX Reader  
   ├── Web content → LlamaIndex Web Reader
   └── Build Vector Index → FAISS

2. Create Analysis Crew (CrewAI)
   ├── Research Agent → Searches RAG
   ├── Data Analyst → Analyzes findings
   └── Report Writer → Synthesizes results

3. Run Analysis
   └── Agents collaborate, query RAG, produce report
```

## 💡 Alternative Options (Not Recommended)

**Plain Implementation:**

-   ❌ Would need to build all document parsers yourself
-   ❌ No agent orchestration framework
-   ❌ 1000+ lines of code

**LangChain Only:**

-   ❌ Weaker document parsing than LlamaIndex
-   ✅ Good agent support
-   ⚠️ CrewAI already uses LangChain

**LlamaIndex Only:**

-   ✅ Excellent RAG
-   ❌ No multi-agent framework
-   ❌ Would need to build agent orchestration

## 📈 Deployment Recommendation

**For VPS setup:**

1.  **Build Phase** (run once or when docs change):
    -   Parse all documents
    -   Build LlamaIndex index
    -   Persist to disk (~1-2GB)
2.  **Query Phase** (run analyses):
    -   Load index from disk (fast)
    -   Initialize CrewAI agents
    -   Run analysis workflows
3.  **Optional: Gradio Interface**:
    -   Let users trigger analyses
    -   View agent progress in real-time
    -   Download reports



---

## 🚀 Get Started
👉 Check [docs/onboarding.md](docs/onboarding.md) and claim your first issue here: [GitHub Issues](../../issues)

Happy Hacking! 💻✨

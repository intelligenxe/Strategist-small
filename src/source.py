"""
Hybrid Architecture: LlamaIndex (RAG) + LangChain (CrewAI Integration)

This architecture leverages:
- LlamaIndex: Superior document parsing and RAG
- LangChain: CrewAI integration and agent orchestration
- Groq: Fast LLM inference
"""

import os
from typing import List, Dict
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from langchain_groq import ChatGroq
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as LlamaGroq
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core.node_parser import SimpleNodeParser


# ============================================================================
# PART 1: RAG KNOWLEDGE BASE BUILDER (LlamaIndex)
# ============================================================================

class KnowledgeBaseBuilder:
    """
    Handles complex document parsing from multiple sources using LlamaIndex
    """
    
    def __init__(self, groq_api_key: str):
        # Configure LlamaIndex settings
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        Settings.llm = LlamaGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_api_key
        )
        
        # Initialize readers for different document types
        self.pdf_reader = PDFReader()
        self.docx_reader = DocxReader()
        
        self.documents = []
        self.index = None
    
    def load_pdfs(self, pdf_paths: List[str]):
        """Parse PDFs with table and image support"""
        for path in pdf_paths:
            docs = self.pdf_reader.load_data(file=path)
            self.documents.extend(docs)
        print(f"Loaded {len(pdf_paths)} PDF files")
    
    def load_docx(self, docx_paths: List[str]):
        """Parse Word documents"""
        for path in docx_paths:
            docs = self.docx_reader.load_data(file=path)
            self.documents.extend(docs)
        print(f"Loaded {len(docx_paths)} DOCX files")
    
    def load_text_files(self, txt_paths: List[str]):
        """Load plain text files"""
        for path in txt_paths:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc = Document(text=content, metadata={"source": path})
                self.documents.append(doc)
        print(f"Loaded {len(txt_paths)} text files")
    
    def add_web_content(self, urls: List[str]):
        """Add web content to knowledge base"""
        from llama_index.readers.web import SimpleWebPageReader
        
        loader = SimpleWebPageReader()
        docs = loader.load_data(urls)
        self.documents.extend(docs)
        print(f"Loaded {len(urls)} web pages")
    
    def build_index(self, persist_dir: str = "./storage"):
        """Build vector index from all documents"""
        print(f"Building index from {len(self.documents)} documents...")
        
        # Parse documents into nodes
        parser = SimpleNodeParser.from_defaults(
            chunk_size=512,
            chunk_overlap=50
        )
        nodes = parser.get_nodes_from_documents(self.documents)
        
        # Create index
        self.index = VectorStoreIndex(nodes)
        
        # Persist to disk
        self.index.storage_context.persist(persist_dir=persist_dir)
        print(f"Index built and saved to {persist_dir}")
        
        return self.index
    
    def load_index(self, persist_dir: str = "./storage"):
        """Load existing index from disk"""
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        self.index = load_index_from_storage(storage_context)
        print("Index loaded from disk")
        return self.index
    
    def get_query_engine(self, similarity_top_k: int = 5):
        """Get query engine for RAG"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        return self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode="compact"
        )


# ============================================================================
# PART 2: RAG TOOL FOR CREWAI (LangChain Bridge)
# ============================================================================

# Global query engine (initialized after building knowledge base)
query_engine = None

@tool("Knowledge Base Search")
def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for relevant information.
    Use this tool to retrieve context from documents before analysis.
    
    Args:
        query: The search query or question
    
    Returns:
        Relevant information from the knowledge base
    """
    if query_engine is None:
        return "Error: Knowledge base not initialized"
    
    response = query_engine.query(query)
    return str(response)


# ============================================================================
# PART 3: CREWAI AGENTS (LangChain)
# ============================================================================

class AnalysisCrewBuilder:
    """
    Build CrewAI agents that use the RAG knowledge base
    """
    
    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_api_key,
            temperature=0.7
        )
    
    def create_research_agent(self) -> Agent:
        """Agent that researches topics using the knowledge base"""
        return Agent(
            role="Research Analyst",
            goal="Extract and synthesize relevant information from the knowledge base",
            backstory="""You are an expert research analyst with deep experience 
            in information retrieval and synthesis. You excel at finding relevant 
            information and presenting it clearly.""",
            tools=[search_knowledge_base],
            llm=self.llm,
            verbose=True
        )
    
    def create_data_analyst_agent(self) -> Agent:
        """Agent that performs data analysis"""
        return Agent(
            role="Data Analyst",
            goal="Analyze data and identify patterns, trends, and insights",
            backstory="""You are a skilled data analyst who can interpret 
            complex information and extract meaningful insights. You use the 
            knowledge base to support your analysis.""",
            tools=[search_knowledge_base],
            llm=self.llm,
            verbose=True
        )
    
    def create_report_writer_agent(self) -> Agent:
        """Agent that writes comprehensive reports"""
        return Agent(
            role="Report Writer",
            goal="Create clear, comprehensive reports based on research and analysis",
            backstory="""You are an expert technical writer who creates 
            well-structured, insightful reports. You synthesize information 
            from multiple sources into coherent narratives.""",
            tools=[search_knowledge_base],
            llm=self.llm,
            verbose=True
        )
    
    def create_analysis_crew(self, research_topic: str) -> Crew:
        """Create a crew for comprehensive analysis"""
        
        # Create agents
        researcher = self.create_research_agent()
        analyst = self.create_data_analyst_agent()
        writer = self.create_report_writer_agent()
        
        # Create tasks
        research_task = Task(
            description=f"""Research the following topic using the knowledge base:
            {research_topic}
            
            Find all relevant information, key facts, and important details.""",
            agent=researcher,
            expected_output="Comprehensive research findings with key information"
        )
        
        analysis_task = Task(
            description="""Analyze the research findings to identify:
            - Key patterns and trends
            - Important insights
            - Potential implications
            - Areas requiring attention
            
            Use the knowledge base to support your analysis.""",
            agent=analyst,
            expected_output="Detailed analysis with insights and patterns",
            context=[research_task]
        )
        
        report_task = Task(
            description="""Create a comprehensive report that includes:
            1. Executive Summary
            2. Key Findings
            3. Detailed Analysis
            4. Conclusions and Recommendations
            
            The report should be well-structured and professional.""",
            agent=writer,
            expected_output="Professional report with all sections",
            context=[research_task, analysis_task]
        )
        
        # Create crew
        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, report_task],
            process=Process.sequential,
            verbose=True
        )
        
        return crew


# ============================================================================
# PART 4: MAIN WORKFLOW
# ============================================================================

def main():
    """
    Main workflow: Build knowledge base, then run analysis
    """
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # ========================================================================
    # STEP 1: Build Knowledge Base (Do once, or when documents change)
    # ========================================================================
    
    print("=" * 70)
    print("STEP 1: Building Knowledge Base")
    print("=" * 70)
    
    kb_builder = KnowledgeBaseBuilder(GROQ_API_KEY)
    
    # Load documents from multiple sources
    kb_builder.load_pdfs([
        "./data/report1.pdf",
        "./data/analysis.pdf"
    ])
    
    kb_builder.load_docx([
        "./data/document1.docx"
    ])
    
    kb_builder.load_text_files([
        "./data/notes.txt",
        "./data/summary.txt"
    ])
    
    kb_builder.add_web_content([
        "https://example.com/article1",
        "https://example.com/article2"
    ])
    
    # Build and persist index
    kb_builder.build_index(persist_dir="./storage")
    
    # Or load existing index:
    # kb_builder.load_index(persist_dir="./storage")
    
    # Set global query engine for CrewAI tools
    global query_engine
    query_engine = kb_builder.get_query_engine(similarity_top_k=5)
    
    # ========================================================================
    # STEP 2: Run CrewAI Analysis
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("STEP 2: Running CrewAI Analysis")
    print("=" * 70)
    
    crew_builder = AnalysisCrewBuilder(GROQ_API_KEY)
    
    # Define your research topic
    research_topic = """
    Analyze the key trends and patterns in the documents related to 
    [YOUR SPECIFIC TOPIC]. Identify the main themes, important findings, 
    and provide strategic recommendations.
    """
    
    # Create and run crew
    crew = crew_builder.create_analysis_crew(research_topic)
    result = crew.kickoff()
    
    # ========================================================================
    # STEP 3: Output Results
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(result)
    
    # Save report
    with open("analysis_report.md", "w") as f:
        f.write(str(result))
    
    print("\nReport saved to analysis_report.md")


if __name__ == "__main__":
    main()


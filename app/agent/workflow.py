import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from app.agent.state import AgentState
from app.agent.tools.vector import VectorStore
from app.agent.tools.graph import KnowledgeGraph
from app.agent.tools.web import WebSearch

# Load environment variables
load_dotenv()

# Initialize Tools
vector_store = VectorStore()
knowledge_graph = KnowledgeGraph()
web_search = WebSearch()

# Initialize LLM (OpenRouter)
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"),
)

# --- Nodes ---

def interpret_query(state: AgentState):
    """Analyze user intent and decide which source to use."""
    print("--- INTERPRET QUERY ---")
    query = state["query"]
    prompt = ChatPromptTemplate.from_template(
        """Analyze the following query and decide the best data source.
        
        Query: {query}
        
        Rules:
        - "vector": For historical facts, long reference text, or specific documents.
        - "graph": For entity relationships, causal links, or multi-hop reasoning.
        - "web": For latest news, current events, or unknown facts.
        - "mixed": If multiple sources are needed.
        
        Return ONLY one word: vector, graph, web, or mixed.
        """
    )
    chain = prompt | llm | StrOutputParser()
    source = chain.invoke({"query": query}).strip().lower()
    print(f"Source: {source}")
    return {"source_selection": source, "plan": [f"Selected source: {source}"]}

def retrieve(state: AgentState):
    """Retrieve data based on source selection."""
    print("--- RETRIEVE ---")
    source = state["source_selection"]
    query = state["query"]
    retrieved_docs = []
    
    if source in ["vector", "mixed"]:
        docs = vector_store.search(query)
        retrieved_docs.extend([d.page_content for d in docs])
        
    if source in ["web", "mixed"]:
        web_results = web_search.search(query)
        retrieved_docs.append(web_results)
        
    if source in ["graph", "mixed"]:
        # Simple graph search for now
        nodes = knowledge_graph.search_nodes(query)
        if nodes:
            subgraph = knowledge_graph.get_subgraph(nodes)
            retrieved_docs.append(f"Graph Data: {subgraph}")
    
    print(f"Retrieved {len(retrieved_docs)} docs")
    return {"retrieved_docs": retrieved_docs}

def extract_knowledge(state: AgentState):
    """Extract entities and relations from retrieved text to update the graph."""
    print("--- EXTRACT KNOWLEDGE ---")
    docs = state["retrieved_docs"]
    if not docs:
        return {}
        
    text_content = "\n".join(docs[:3]) # Limit to avoid token limits
    
    prompt = ChatPromptTemplate.from_template(
        """Extract knowledge triplets from the following text.
        Format: Subject | Predicate | Object
        
        Text: {text}
        
        Return ONLY the triplets, one per line.
        """
    )
    chain = prompt | llm | StrOutputParser()
    triplets_str = chain.invoke({"text": text_content})
    
    # Parse and update graph
    new_relations = []
    for line in triplets_str.split("\n"):
        parts = line.split("|")
        if len(parts) == 3:
            s, p, o = [x.strip() for x in parts]
            knowledge_graph.add_relation(s, p, o)
            new_relations.append({"subject": s, "predicate": p, "object": o})
            
    return {"knowledge_graph": new_relations}

def generate_answer(state: AgentState):
    """Synthesize the final answer."""
    print("--- GENERATE ANSWER ---")
    query = state["query"]
    docs = state["retrieved_docs"]
    
    context = "\n\n".join(docs)
    
    prompt = ChatPromptTemplate.from_template(
        """Answer the user query based on the provided context.
        
        Query: {query}
        
        Context:
        {context}
        
        Requirements:
        - Cite sources (Vector/KG/Web).
        - Show reasoning.
        - Be concise and grounded.
        """
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"query": query, "context": context})
    
    print(f"Answer: {answer[:50]}...")
    return {"final_answer": answer}

# --- Graph Definition ---

workflow = StateGraph(AgentState)

workflow.add_node("interpret_query", interpret_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("extract_knowledge", extract_knowledge)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("interpret_query")

workflow.add_edge("interpret_query", "retrieve")
workflow.add_edge("retrieve", "extract_knowledge")
workflow.add_edge("extract_knowledge", "generate_answer")
workflow.add_edge("generate_answer", END)

memory = MemorySaver()
app_workflow = workflow.compile(checkpointer=memory)

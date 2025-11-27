from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    query: str
    messages: List[Dict[str, str]]
    retrieved_docs: List[str]
    knowledge_graph: List[Dict[str, Any]] # List of triplets or graph data
    plan: List[str]
    final_answer: Optional[str]
    source_selection: str # "vector", "graph", "web", or "mixed"

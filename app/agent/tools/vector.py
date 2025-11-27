import os
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

class VectorStore:
    def __init__(self, collection_name: str = "agent_memory"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
        # Initialize BM25 retriever (will be empty initially or loaded if we had persistence logic)
        # For simplicity in this demo, we'll rebuild BM25 from Chroma docs if possible, 
        # or just keep it empty until docs are added.
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self._initialize_retrievers()

    def _initialize_retrievers(self):
        """Initialize or update retrievers."""
        try:
            # Get all docs from Chroma to initialize BM25
            # Note: This is expensive for large DBs, but fine for a demo.
            existing_docs = self.vector_db.get()
            if existing_docs['documents']:
                docs = [
                    Document(page_content=t, metadata=m) 
                    for t, m in zip(existing_docs['documents'], existing_docs['metadatas'])
                ]
                self.bm25_retriever = BM25Retriever.from_documents(docs)
                self.bm25_retriever.k = 5
                
                chroma_retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})
                
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.bm25_retriever, chroma_retriever],
                    weights=[0.5, 0.5]
                )
            else:
                self.bm25_retriever = None
                self.ensemble_retriever = None
        except Exception as e:
            print(f"Error initializing retrievers: {e}")

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add texts to the vector store and update retrievers."""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        docs = [
            Document(page_content=t, metadata=m) 
            for t, m in zip(texts, metadatas)
        ]
        self.vector_db.add_documents(docs)
        self._initialize_retrievers() # Re-initialize to include new docs in BM25

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search using Ensemble Retriever (Hybrid) or fallback to Vector."""
        if self.ensemble_retriever:
            return self.ensemble_retriever.invoke(query)
        else:
            return self.vector_db.similarity_search(query, k=k)

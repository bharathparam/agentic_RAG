from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import shutil
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader

load_dotenv()

app = FastAPI(title="AGENT-TRI-RAG", version="1.0")

# Initialize Vector Store globally for the app
from app.agent.tools.vector import VectorStore
vector_store = VectorStore()

# Import workflow
from app.agent.workflow import app_workflow

class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = None
    thread_id: Optional[str] = "default_thread"

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "AGENT-TRI-RAG"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF or Text) to the RAG system."""
    try:
        # Save file temporarily
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load document
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            texts = [d.page_content for d in documents]
            metadatas = [d.metadata for d in documents]
        else:
            # Assume text
            loader = TextLoader(file_path)
            documents = loader.load()
            texts = [d.page_content for d in documents]
            metadatas = [{"source": file.filename} for _ in texts]
            
        # Add to Vector Store (Hybrid)
        vector_store.add_documents(texts, metadatas)
        
        # Cleanup
        os.remove(file_path)
        
        return {"status": "success", "filename": file.filename, "chunks_added": len(texts)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def run_agent(request: QueryRequest):
    initial_state = {
        "query": request.query,
        "messages": [],
        "retrieved_docs": [],
        "knowledge_graph": [],
        "plan": [],
        "final_answer": None,
        "source_selection": ""
    }
    
    config = {"configurable": {"thread_id": request.thread_id}}
    
    try:
        result = app_workflow.invoke(initial_state, config=config)
        
        return {
            "query": request.query,
            "response": result.get("final_answer", "No answer generated."),
            "sources": result.get("source_selection", "unknown"),
            "plan": result.get("plan", []),
            "thread_id": request.thread_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

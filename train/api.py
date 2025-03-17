"""
API for query reformulation.
"""
import time
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn

from models import load_model, generate_reformulation, MODEL_SIZES

app = FastAPI(title="Query Reformulation API")

# Global variables to store model, tokenizer, and device
model = None
tokenizer = None
device = None
model_info = {"size": "base", "loaded": False}

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    reformulation: str
    time_ms: float

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, tokenizer, device, model_info
    model_size = model_info["size"]
    print(f"Loading Flan-T5-{model_size} model...")
    model, tokenizer, device = load_model(model_size, force_cpu=True)
    model_info["loaded"] = True
    print("Model loaded successfully!")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Query Reformulation API", "model": f"flan-t5-{model_info['size']}"}

@app.post("/reformulate", response_model=QueryResponse)
async def reformulate_query(request: QueryRequest):
    """Reformulate a query."""
    if not model_info["loaded"]:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    
    start_time = time.time()
    reformulation = generate_reformulation(model, tokenizer, request.query, device)
    elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return QueryResponse(reformulation=reformulation, time_ms=elapsed_time)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="API for query reformulation")
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="base", 
                        help="Size of the model (small, base, large)")
    parser.add_argument("--host", type=str, default="127.0.0.1", 
                        help="Host to run the API on")
    parser.add_argument("--port", type=int, default=8000, 
                        help="Port to run the API on")
    
    args = parser.parse_args()
    
    # Set model size before startup
    model_info["size"] = args.model_size
    
    uvicorn.run(app, host=args.host, port=args.port)

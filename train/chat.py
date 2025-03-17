"""
Interactive chat interface for query reformulation.
"""
import time
from typing import List, Dict, Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from models import load_model, generate_reformulation, MODEL_SIZES

def chat_loop(model_size: str = "base", force_cpu: bool = False) -> None:
    """Run an interactive chat loop for query reformulation.
    
    Args:
        model_size: Size of the model ('small', 'base', or 'large')
        force_cpu: Whether to force CPU usage
    """
    print(f"Loading Flan-T5-{model_size} model...")
    model, tokenizer, device = load_model(model_size, force_cpu)
    
    print("\n" + "=" * 50)
    print("Query Reformulation Chat")
    print("Type 'exit' or 'quit' to end the session")
    print("=" * 50 + "\n")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ["exit", "quit"]:
            break
            
        start_time = time.time()
        reformulation = generate_reformulation(model, tokenizer, query, device)
        elapsed_time = time.time() - start_time
        
        print(f"\n🔄 Reformulated query: {reformulation}")
        print(f"⏱️ Time taken: {elapsed_time*1000:.2f} ms")
    
    print("\nThank you for using the Query Reformulation Chat!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive chat for query reformulation")
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="base", 
                        help="Size of the model (small, base, large)")
    parser.add_argument("--force-cpu", action="store_true", 
                        help="Force using CPU even if GPU/MPS is available")
    
    args = parser.parse_args()
    
    chat_loop(args.model_size, args.force_cpu)

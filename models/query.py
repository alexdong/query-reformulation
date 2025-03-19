import torch 
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import List, Optional, Union

class QueryReformulator:
    def __init__(self, model_size: str = "small", use_sft_model: bool = True):
        """Initialize the query reformulation model.
        
        Args:
            model_size: Size of the T5 model ('small', 'base', or 'large')
            use_sft_model: Whether to use the fine-tuned model
        """
        self.model_size = model_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the appropriate model
        model_path = f"./models/sft-{model_size}" if use_sft_model else f"t5-{model_size}"
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def reformulate(self, query: str, max_length: int = 128, num_return_sequences: int = 1) -> List[str]:
        """Reformulate a query into one or more search engine queries.
        
        Args:
            query: The input query to reformulate
            max_length: Maximum length of the generated output
            num_return_sequences: Number of reformulations to generate
            
        Returns:
            A list of reformulated queries
        """
        # Prepare input
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                num_beams=num_return_sequences * 2,
                early_stopping=True
            )
        
        # Decode and return reformulated queries
        reformulated_queries = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return reformulated_queries

def reformulate_query(query: str, model_size: str = "small", max_length: int = 128) -> List[str]:
    """Convenience function to reformulate a query without explicitly creating a QueryReformulator.
    
    Args:
        query: The input query to reformulate
        model_size: Size of the T5 model ('small', 'base', or 'large')
        max_length: Maximum length of the generated output
        
    Returns:
        A list of reformulated queries
    """
    reformulator = QueryReformulator(model_size=model_size)
    return reformulator.reformulate(query, max_length=max_length)

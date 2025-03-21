import argparse
import torch
import logging
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from torch.nn import Linear

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class E5SmolLMInference:
    def __init__(self, e5_model_name, checkpoint_dir):
        """
        Initialize the E5SmolLM inference model.
        
        Args:
            e5_model_name: The name of the E5 embedding model
            checkpoint_dir: Directory containing the fine-tuned SmolLM checkpoint
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load E5 embedding model
        logger.info(f"Loading E5 embedding model: {e5_model_name}")
        self.e5_tokenizer = AutoTokenizer.from_pretrained(e5_model_name)
        self.e5_model = AutoModel.from_pretrained(e5_model_name)
        
        # Set padding token if not set
        if self.e5_tokenizer.pad_token is None:
            if self.e5_tokenizer.eos_token is not None:
                self.e5_tokenizer.pad_token = self.e5_tokenizer.eos_token
            else:
                self.e5_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
        # Load SmolLM model from checkpoint
        logger.info(f"Loading SmolLM model from checkpoint: {checkpoint_dir}")
        # self.lm_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        self.lm_tokenizer = AutoTokenizer.from_pretrained(e5_model_name)
        self.lm_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            device_map=self.device
        )
        
        # Freeze E5 parameters
        for param in self.e5_model.parameters():
            param.requires_grad = False
        
        # Get embedding dimensions
        self.e5_embedding_dim = self.e5_model.config.hidden_size
        self.lm_embedding_dim = self.lm_model.config.hidden_size
        
        logger.info(f"E5 embedding dimension: {self.e5_embedding_dim}")
        logger.info(f"LM embedding dimension: {self.lm_embedding_dim}")
        
        # Create projection layer if dimensions don't match
        if self.e5_embedding_dim != self.lm_embedding_dim:
            logger.info(f"Creating projection layer from {self.e5_embedding_dim} to {self.lm_embedding_dim}")
            self.projection = Linear(self.e5_embedding_dim, self.lm_embedding_dim)
        else:
            self.projection = torch.nn.Identity()
            
        # Move models to device
        self.e5_model.to(self.device)
        self.lm_model.to(self.device)
        if hasattr(self, 'projection') and not isinstance(self.projection, torch.nn.Identity):
            self.projection.to(self.device)
            
    def generate(self, text, max_length=50, num_return_sequences=1, temperature=0.7, 
                 top_p=0.9, repetition_penalty=1.0, do_sample=True):
        """
        Generate text using the E5-SmolLM model.
        
        Args:
            text: Input text prompt
            max_length: Maximum length of generated text
            num_return_sequences: Number of sequences to return
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeated tokens
            do_sample: Whether to use sampling instead of greedy decoding
            
        Returns:
            List of generated text sequences
        """
        # Tokenize input text
        e5_inputs = self.e5_tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate embeddings with E5
        with torch.no_grad():
            e5_outputs = self.e5_model(**e5_inputs, return_dict=True)
            embeddings = e5_outputs.last_hidden_state
            
            # Project embeddings if necessary
            projected_embeddings = self.projection(embeddings)
            
            # Generate text using SmolLM
            outputs = self.lm_model.generate(
                inputs_embeds=projected_embeddings,
                attention_mask=e5_inputs['attention_mask'],
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.lm_tokenizer.pad_token_id,
                eos_token_id=self.lm_tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Decode generated token IDs
            generated_text = self.lm_tokenizer.batch_decode(
                outputs.sequences, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return generated_text

def main():
    parser = argparse.ArgumentParser(description="Inference with E5-SmolLM model")
    
    # Model arguments
    parser.add_argument("--e5_model_name", type=str, default="intfloat/multilingual-e5-large", 
                        help="E5 model name or path")
    parser.add_argument("--checkpoint_dir", type=str, default="./e5-smollm-model", 
                        help="Directory containing the fine-tuned SmolLM checkpoint")
    
    # Generation arguments
    parser.add_argument("--input_text", type=str, required=True,
                        help="Input text for text generation")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated text")
    parser.add_argument("--num_sequences", type=int, default=1,
                        help="Number of sequences to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Penalty for repeated tokens")
    parser.add_argument("--no_sample", action="store_true",
                        help="Use greedy decoding instead of sampling")
    
    args = parser.parse_args()
    
    # Initialize model
    model = E5SmolLMInference(
        e5_model_name=args.e5_model_name,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Generate text
    generated_texts = model.generate(
        text=args.input_text,
        max_length=args.max_length,
        num_return_sequences=args.num_sequences,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=not args.no_sample
    )
    
    # Print generated text
    print("\n" + "="*50)
    print(f"Input: {args.input_text}")
    print("="*50)
    for i, text in enumerate(generated_texts):
        print(f"\nGenerated text {i+1}:")
        print("-"*50)
        print(text)
    print("="*50)


if __name__ == "__main__":
    main()
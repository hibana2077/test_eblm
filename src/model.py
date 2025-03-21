import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import load_dataset
import logging
import argparse
from tqdm import tqdm

class E5SmolLMModel(nn.Module):
    def __init__(self, e5_model_name, lm_model_name):
        super().__init__()
        # Load E5 embedding model
        print(f"Loading E5 embedding model: {e5_model_name}")
        self.e5_tokenizer = AutoTokenizer.from_pretrained(e5_model_name)
        self.e5_model = AutoModel.from_pretrained(e5_model_name)

        # Freeze E5 parameters
        for param in self.e5_model.parameters():
            param.requires_grad = False
            
        # Load SmolLM2 model
        print(f"Loading SmolLM2 model: {lm_model_name}")
        self.lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        self.lm_model = AutoModelForCausalLM.from_pretrained(lm_model_name)
        
        # Get embedding dimensions
        e5_embedding_dim = self.e5_model.config.hidden_size
        lm_embedding_dim = self.lm_model.config.hidden_size
        
        print(f"E5 embedding dimension: {e5_embedding_dim}")
        print(f"LM embedding dimension: {lm_embedding_dim}")
        
        # Create projection layer if dimensions don't match
        if e5_embedding_dim != lm_embedding_dim:
            print(f"Creating projection layer from {e5_embedding_dim} to {lm_embedding_dim}")
            self.projection = nn.Linear(e5_embedding_dim, lm_embedding_dim)
        else:
            self.projection = nn.Identity()
            
        # Remove and replace the embedding layer from SmolLM2
        self.original_word_embeddings = self.lm_model.get_input_embeddings()
        
        # Resize the token embeddings to match E5 tokenizer size
        self.lm_model.resize_token_embeddings(self.e5_tokenizer.vocab_size) # recommended


    def forward(self, input_ids, attention_mask, labels=None):
        # Get E5 embeddings
        e5_outputs = self.e5_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            return_dict=True
        )
        # print(f"E5 outputs: {e5_outputs.last_hidden_state.shape}")
        # Use the last hidden state as embeddings
        embeddings = e5_outputs.last_hidden_state
        
        # Project embeddings if necessary
        projected_embeddings = self.projection(embeddings)
        
        # Pass through SmolLM2 but bypass the embedding layer
        outputs = self.lm_model(
            inputs_embeds=projected_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
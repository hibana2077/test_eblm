import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import load_dataset
import logging
import argparse
from tqdm import tqdm

from model import E5SmolLMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, dataset, e5_tokenizer, lm_tokenizer, max_length=512):
        self.dataset = dataset
        self.e5_tokenizer = e5_tokenizer
        self.e5_tokenizer.pad_token = self.e5_tokenizer.eos_token
        self.lm_tokenizer = lm_tokenizer
        self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        
        # Tokenize for E5
        e5_encodings = self.e5_tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize for LM (for labels)
        lm_encodings = self.lm_tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'e5_input_ids': e5_encodings.input_ids.flatten(),
            'e5_attention_mask': e5_encodings.attention_mask.flatten(),
            'labels': lm_encodings.input_ids.flatten()
        }


def train(args):
    # Initialize models
    model = E5SmolLMModel(args.e5_model_name, args.lm_model_name)
    model.to(args.device)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    # dataset = load_dataset(args.dataset_name, split="train[:10%]")  # Using a small portion for MVP
    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split="train[:10%]")  # Using a small portion for MVP
    
    # Create dataset and dataloader
    train_dataset = TextDataset(
        dataset, 
        model.e5_tokenizer, 
        model.lm_tokenizer, 
        max_length=args.max_length
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )
    
    # Learning rate scheduler
    num_training_steps = len(train_dataloader) * args.num_epochs
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    logger.info("Starting training")
    model.train()
    
    for epoch in range(args.num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            e5_input_ids = batch['e5_input_ids'].to(args.device)
            e5_attention_mask = batch['e5_attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=e5_input_ids,
                attention_mask=e5_attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 
                args.max_grad_norm
            )
            
            # Update parameters
            optimizer.step()
            lr_scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})
        
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}: Average loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = f"{args.output_dir}/checkpoint-{epoch + 1}"
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            model.lm_model.save_pretrained(checkpoint_path)
    
    # Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    model.lm_model.save_pretrained(args.output_dir)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train a model with E5 embeddings and SmolLM2")
    
    # Model arguments
    parser.add_argument("--e5_model_name", type=str, default="intfloat/multilingual-e5-large", 
                        help="E5 model name or path")
    parser.add_argument("--lm_model_name", type=str, default="HuggingFaceTB/SmolLM2-135M", 
                        help="SmolLM2 model name or path")
    
    # Training arguments
    parser.add_argument("--dataset_name", type=str, default="wikitext", 
                        help="Dataset to use for training")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1",
                        help="Dataset configuration name")
    parser.add_argument("--output_dir", type=str, default="./e5-smollm-model", 
                        help="Directory to save the model")
    parser.add_argument("--max_length", type=int, default=512, 
                        help="Maximum length of input sequences")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=500, 
                        help="Number of warmup steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", 
                        help="Type of learning rate scheduler")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, 
                        help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--save_every", type=int, default=1, 
                        help="Save checkpoint every n epochs")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for training")
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()
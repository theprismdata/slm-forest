#!/usr/bin/env python3
"""
Phi-2 Fine-tuning Script for Tax Law Q&A Dataset
Optimized for RunPod environment with LoRA
"""

import os
import json
import logging
import yaml
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    model_name: str = "microsoft/phi-2"
    max_length: int = 1024
    padding: str = "max_length"
    truncation: bool = True
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "./phi2-tax-law-finetuned"
    fp16: bool = True
    report_to: str = "none"
    run_name: str = "phi2-tax-law-qa"
    
    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Dataset config
    train_file: str = "fine-tunning-ds/distillation_legal_qa_dataset.json"
    validation_split: float = 0.1
    
    # Prompt template
    prompt_template: str = "### 질문: {question}\n\n### 답변: {answer}"

def load_config(config_path: str = "train_config.yaml") -> TrainingConfig:
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract training args
        training_args = config_dict.get('training_args', {})
        lora_config = config_dict.get('lora_config', {})
        dataset_config = config_dict.get('dataset', {})
        
        return TrainingConfig(
            model_name=config_dict.get('model_name', 'microsoft/phi-2'),
            max_length=config_dict.get('max_length', 1024),
            padding=config_dict.get('padding', 'max_length'),
            truncation=config_dict.get('truncation', True),
            num_train_epochs=training_args.get('num_train_epochs', 3),
            per_device_train_batch_size=training_args.get('per_device_train_batch_size', 2),
            per_device_eval_batch_size=training_args.get('per_device_eval_batch_size', 2),
            gradient_accumulation_steps=training_args.get('gradient_accumulation_steps', 8),
            learning_rate=float(training_args.get('learning_rate', 2e-5)),
            weight_decay=float(training_args.get('weight_decay', 0.01)),
            warmup_steps=training_args.get('warmup_steps', 100),
            logging_steps=training_args.get('logging_steps', 10),
            save_steps=training_args.get('save_steps', 500),
            eval_steps=training_args.get('eval_steps', 500),
            output_dir=training_args.get('output_dir', './phi2-tax-law-finetuned'),
            fp16=training_args.get('fp16', True),
            report_to=training_args.get('report_to', 'none'),
            run_name=training_args.get('run_name', 'phi2-tax-law-qa'),
            lora_r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            lora_dropout=float(lora_config.get('lora_dropout', 0.1)),
            train_file=dataset_config.get('train_file', 'fine-tunning-ds/distillation_legal_qa_dataset.json'),
            validation_split=float(dataset_config.get('validation_split', 0.1)),
            prompt_template=config_dict.get('prompt_template', "### 질문: {question}\n\n### 답변: {answer}")
        )
    else:
        logger.warning(f"Config file {config_path} not found, using default config")
        return TrainingConfig()

def load_dataset(file_path: str) -> List[Dict]:
    """Load dataset from JSON file"""
    logger.info(f"Loading dataset from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} samples from dataset")
    return data

def prepare_dataset(data: List[Dict], config: TrainingConfig) -> tuple[Dataset, Dataset]:
    """Prepare dataset for training"""
    logger.info("Preparing dataset for training")
    
    # Format data with prompt template
    formatted_data = []
    for item in data:
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        # Format with prompt template
        formatted_text = config.prompt_template.format(
            question=question,
            answer=answer
        )
        
        formatted_data.append({
            'text': formatted_text,
            'question': question,
            'answer': answer,
            'type': item.get('type', 'student'),
            'source': item.get('source', 'practical')
        })
    
    # Split into train and validation
    train_data, val_data = train_test_split(
        formatted_data, 
        test_size=config.validation_split, 
        random_state=42
    )
    
    logger.info(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    return Dataset.from_list(train_data), Dataset.from_list(val_data)

def tokenize_function(examples, tokenizer, config):
    """Tokenize function for dataset"""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding=config.padding,
        max_length=config.max_length,
        return_tensors="pt"
    )

def main():
    """Main training function"""
    logger.info("Starting Phi-2 fine-tuning for Tax Law Q&A")
    
    # Load configuration
    config = load_config()
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"Loading model from {config.model_name}")
    torch.cuda.empty_cache()  # Clear CUDA cache before model loading
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    
    # Enable gradient checkpointing
    model.config.use_cache = False  # Required for gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    logger.info("Configuring LoRA")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    raw_data = load_dataset(config.train_file)
    train_dataset, val_dataset = prepare_dataset(raw_data, config)
    
    # Tokenize datasets
    logger.info("Tokenizing datasets")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        bf16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=config.report_to,
        run_name=config.run_name,
        save_total_limit=3,
        logging_dir=f"{config.output_dir}/logs",
        gradient_checkpointing=True,
        optim="paged_adamw_32bit"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Save training config
    with open(f"{config.output_dir}/training_config.json", 'w', encoding='utf-8') as f:
        json.dump(config.__dict__, f, indent=2, ensure_ascii=False)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 
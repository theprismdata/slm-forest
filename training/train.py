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
import sys
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import deepspeed
from accelerate import Accelerator
import wandb
from tqdm.auto import tqdm

from create_model.model import GemmaForCausalLM, ModelConfig
from slm_datagen.data_pipeline import CustomTokenizer

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

class TrainingArguments:
    """훈련 설정"""
    def __init__(
        self,
        output_dir: str = "outputs",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.1,
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 1000,
        local_rank: int = -1,
        fp16: bool = True,
        bf16: bool = False,
        deepspeed_config: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.local_rank = local_rank
        self.fp16 = fp16
        self.bf16 = bf16
        self.deepspeed_config = deepspeed_config

class Trainer:
    def __init__(
        self,
        model: GemmaForCausalLM,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[DataCollatorForLanguageModeling] = None,
    ):
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Accelerator 초기화
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="fp16" if args.fp16 else "bf16" if args.bf16 else "no",
        )
        
        # 데이터 로더 설정
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )
        
        if eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=data_collator,
            )
        
        # 옵티마이저 설정
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        
        # 학습률 스케줄러 설정
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        num_warmup_steps = int(max_train_steps * args.warmup_ratio)
        
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
        
        # DeepSpeed 설정 (A100 클러스터용)
        if args.deepspeed_config is not None:
            model, self.optimizer, _, _ = deepspeed.initialize(
                model=model,
                optimizer=self.optimizer,
                config=args.deepspeed_config,
            )
        
        # 모델, 옵티마이저, 데이터로더를 accelerator로 준비
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )
        
        if self.eval_dataset is not None:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
        
        self.completed_steps = 0
        self.max_train_steps = max_train_steps
        
    def train(self):
        """훈련 실행"""
        # wandb 초기화
        if self.accelerator.is_main_process:
            wandb.init(project="slm-forest", name="gemma-style-10b")
        
        progress_bar = tqdm(
            total=self.max_train_steps,
            disable=not self.accelerator.is_local_main_process,
        )
        
        for epoch in range(self.args.num_train_epochs):
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.completed_steps += 1
                
                if self.completed_steps % self.args.logging_steps == 0:
                    if self.accelerator.is_main_process:
                        wandb.log(
                            {
                                "train_loss": loss.item(),
                                "learning_rate": self.optimizer.param_groups[0]["lr"],
                                "epoch": epoch,
                                "step": self.completed_steps,
                            }
                        )
                
                if self.completed_steps % self.args.eval_steps == 0:
                    self.evaluate()
                
                if self.completed_steps % self.args.save_steps == 0:
                    self.save_model()
                
                if self.completed_steps >= self.max_train_steps:
                    break
        
        # 최종 모델 저장
        self.save_model()
    
    def evaluate(self):
        """평가 실행"""
        if self.eval_dataset is None:
            return
        
        self.model.eval()
        eval_loss = 0
        eval_steps = 0
        
        for batch in self.eval_dataloader:
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss
                eval_loss += loss.item()
                eval_steps += 1
        
        eval_loss = eval_loss / eval_steps
        perplexity = math.exp(eval_loss)
        
        if self.accelerator.is_main_process:
            wandb.log(
                {
                    "eval_loss": eval_loss,
                    "perplexity": perplexity,
                    "step": self.completed_steps,
                }
            )
        
        self.model.train()
    
    def save_model(self):
        """모델 저장"""
        if not self.accelerator.is_main_process:
            return
        
        output_dir = os.path.join(
            self.args.output_dir,
            f"checkpoint-{self.completed_steps}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # accelerator로 언래핑된 모델 저장
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
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
        warmup_ratio=0.1,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        local_rank=-1,
        fp16=True,
        bf16=False,
        deepspeed_config="training/ds_config.json"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
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
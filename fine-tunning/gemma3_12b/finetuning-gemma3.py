#!/usr/bin/env python3
"""
Gemma 3 12B Fine-tuning Script for Tax Law Q&A Dataset
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
    AutoProcessor,
    Gemma3ForConditionalGeneration,  # Gemma 3 전용 클래스
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import math
# import torch.distributed as dist  # 필요시에만 사용
# from torch.nn.parallel import DistributedDataParallel  # 필요시에만 사용
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
# import deepspeed  # 제거 - 선택사항
from accelerate import Accelerator
import wandb
from tqdm.auto import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    model_name: str = "google/gemma-3-12b-it"  # 올바른 Gemma 3 12B 모델명
    max_length: int = 2048  # Gemma 3에 맞게 설정
    padding: str = "max_length"
    truncation: bool = True
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1  # 12B 모델이므로 배치 크기 감소
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16  # 배치 크기 감소로 인한 보상
    learning_rate: float = 1e-5  # 큰 모델에 맞게 학습률 감소
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "./gemma3-12b-tax-law-finetuned"
    fp16: bool = True
    report_to: str = "none"
    run_name: str = "gemma3-12b-tax-law-qa"
    
    # LoRA config - 12B 모델에 맞게 조정
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    
    # Dataset config
    train_file: str = "fine-tunning-ds/distillation_legal_qa_dataset.json"  # 윈도우 경로 수정
    validation_split: float = 0.1
    
    # Prompt template - Gemma 3 형식으로 수정
    prompt_template: str = "<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>"

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
            model_name=config_dict.get('model_name', 'google/gemma-3-12b-it'),
            max_length=config_dict.get('max_length', 2048),
            padding=config_dict.get('padding', 'max_length'),
            truncation=config_dict.get('truncation', True),
            num_train_epochs=training_args.get('num_train_epochs', 3),
            per_device_train_batch_size=training_args.get('per_device_train_batch_size', 1),
            per_device_eval_batch_size=training_args.get('per_device_eval_batch_size', 1),
            gradient_accumulation_steps=training_args.get('gradient_accumulation_steps', 16),
            learning_rate=float(training_args.get('learning_rate', 1e-5)),
            weight_decay=float(training_args.get('weight_decay', 0.01)),
            warmup_steps=training_args.get('warmup_steps', 100),
            logging_steps=training_args.get('logging_steps', 10),
            save_steps=training_args.get('save_steps', 500),
            eval_steps=training_args.get('eval_steps', 500),
            output_dir=training_args.get('output_dir', './gemma3-12b-tax-law-finetuned'),
            fp16=training_args.get('fp16', True),
            report_to=training_args.get('report_to', 'none'),
            run_name=training_args.get('run_name', 'gemma3-12b-tax-law-qa'),
            lora_r=lora_config.get('r', 32),
            lora_alpha=lora_config.get('lora_alpha', 64),
            lora_dropout=float(lora_config.get('lora_dropout', 0.1)),
            train_file=dataset_config.get('train_file', 'fine-tunning-ds/distillation_legal_qa_dataset.json'),
            validation_split=float(dataset_config.get('validation_split', 0.1)),
            prompt_template=config_dict.get('prompt_template', "<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>")
        )
    else:
        logger.warning(f"Config file {config_path} not found, using default config")
        return TrainingConfig()

def load_dataset(file_path: str) -> List[Dict]:
    """Load dataset from JSON file"""
    logger.info(f"Loading dataset from {file_path}")
    
    # 윈도우 환경에서 경로 처리
    if not os.path.exists(file_path):
        # 상대 경로로 다시 시도
        alternative_paths = [
            os.path.join("..", "slm_datagen", "fine-tunning-ds", "distillation_legal_qa_dataset.json"),
            os.path.join("slm_datagen", "fine-tunning-ds", "distillation_legal_qa_dataset.json"),
            "distillation_legal_qa_dataset.json"
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                file_path = alt_path
                logger.info(f"Found dataset at alternative path: {file_path}")
                break
        else:
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

def check_system_requirements():
    """시스템 요구사항 확인"""
    logger.info("=== System Requirements Check ===")
    
    # CUDA 확인
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"CUDA available: {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            if gpu_memory < 20:
                logger.warning(f"GPU {i} has less than 20GB memory. Consider using smaller batch size.")
    else:
        logger.error("CUDA not available. GPU training is required for Gemma3 12B.")
        return False
    
    # 메모리 확인
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)
    logger.info(f"System RAM: {ram_gb:.1f}GB")
    
    if ram_gb < 32:
        logger.warning("Less than 32GB RAM. Consider using smaller datasets or more aggressive quantization.")
    
    # 디스크 공간 확인
    disk_free = psutil.disk_usage('.').free / (1024**3)
    logger.info(f"Free disk space: {disk_free:.1f}GB")
    
    if disk_free < 50:
        logger.warning("Less than 50GB free disk space. Model checkpoints may require significant space.")
    
    return True

def main():
    """Main training function"""
    logger.info("Starting Gemma 3 12B fine-tuning for Tax Law Q&A")
    
    # 시스템 요구사항 확인
    if not check_system_requirements():
        logger.error("System requirements not met. Exiting.")
        return
    
    # Load configuration
    config = load_config()
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cpu":
        logger.error("CUDA not available. GPU is required for Gemma 3 12B training.")
        return
    
    # Load processor (Gemma 3는 멀티모달이므로 processor 사용)
    logger.info(f"Loading processor from {config.model_name}")
    try:
        processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)
        tokenizer = processor.tokenizer  # 토크나이저는 프로세서에서 가져옴
    except Exception as e:
        logger.error(f"Failed to load processor: {e}")
        logger.info("Please check if the model name is correct and you have internet access.")
        return
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - Gemma 3 전용 클래스 사용
    logger.info(f"Loading model from {config.model_name}")
    torch.cuda.empty_cache()  # Clear CUDA cache before model loading
    
    try:
        # Configure quantization for 12B model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 12B 모델이므로 4bit 양자화 사용
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        model = Gemma3ForConditionalGeneration.from_pretrained(  # Gemma 3 전용 클래스
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("This might be due to insufficient GPU memory or incorrect model name.")
        return
    
    # Enable gradient checkpointing
    model.config.use_cache = False  # Required for gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA - Gemma 3 모델의 실제 레이어 이름 사용
    logger.info("Configuring LoRA")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Gemma 3 레이어 이름
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    try:
        raw_data = load_dataset(config.train_file)
        train_dataset, val_dataset = prepare_dataset(raw_data, config)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
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
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=config.fp16,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=config.report_to,
        run_name=config.run_name,
        ddp_find_unused_parameters=False,
        group_by_length=True,  # 효율적인 배치 구성
        length_column_name="length",
        disable_tqdm=False,
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
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return
    
    # Save final model
    logger.info("Saving final model")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)  # processor 저장
    
    # Save training config
    with open(f"{config.output_dir}/training_config.json", 'w', encoding='utf-8') as f:
        json.dump(config.__dict__, f, indent=2, ensure_ascii=False)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 
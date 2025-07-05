#!/usr/bin/env python3
"""
Mistral-7B Chat Interface
Simple interactive chat interface using Mistral-7B-Instruct-v0.3 from Hugging Face
Optimized for Apple Silicon (M1/M2) using MPS
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MistralChat:
    def __init__(self):
        """
        Initialize the chatbot with MPS device
        """
        self.device = "mps"  # Fixed to MPS
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        self.tokenizer = None
        self.model = None
        self.chat_history = []
        # Mistral specific prompt template
        self.prompt_template = """<s>[INST] {question} [/INST]"""
        
        logger.info(f"Initializing Mistral-7B from Hugging Face")
        logger.info(f"Using device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer from Hugging Face"""
        try:
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Using float32 for MPS stability
                trust_remote_code=True,
                device_map="auto"
            ).to(self.device)
            
            self.model.eval()
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_response(
        self, 
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ) -> str:
        """Generate response for the given question"""
        try:
            # Format prompt with Mistral's template
            prompt = self.prompt_template.format(question=question)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3
                )
            
            # Decode and clean up the response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"An error occurred: {str(e)}"
    
    def chat(self):
        """Interactive chat mode"""
        print("\n=== Mistral-7B Chat (MPS) ===")
        print("Enter your message. Type 'quit' or 'exit' to end the conversation.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("Thinking...")
                response = self.generate_response(user_input)
                print(f"\nAssistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

def main():
    """Main function"""
    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device is not available. This script requires an Apple Silicon Mac.")
            
        chatbot = MistralChat()
        chatbot.chat()
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")

if __name__ == "__main__":
    main()

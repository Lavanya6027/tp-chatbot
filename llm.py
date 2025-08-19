# llm.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LocalLLM:
    """Interface to any local LLM, using Hugging Face + PyTorch"""
    def __init__(self, model_name_or_path: str, device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device_map="auto" if "cuda" in self.device else None
        )

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

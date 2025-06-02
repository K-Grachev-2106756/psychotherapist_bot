import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
)

from config import hf_token


class LLM:

    def init(model_name: str):
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map="cuda:0",
            trust_remote_code=True,
            token=hf_token, 
        )
        base_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=hf_token, 
            use_fast=True,
        )

        return base_model, tokenizer

    
    def load_lora(base_model: AutoModelForCausalLM, model_path: str):
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()

        return model
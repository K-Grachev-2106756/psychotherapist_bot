import os
from dotenv import load_dotenv


load_dotenv()

token = os.getenv("HF_TOKEN")

model_name = "mistralai/Mistral-7B-Instruct-v0.3"


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import prepare_model_for_kbit_training


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_model(model_name):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, token=token, quantization_config=bnb_cfg)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model.to("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    return tokenizer, model


from peft import PeftModel

tokenizer, model = load_model(model_name)

base_total, base_trainable = count_parameters(model)
print(f"[Base Model] Total: {base_total:,}, Trainable: {base_trainable:,}")

model = PeftModel.from_pretrained(model, "./models/textbook/mistral-psy-lora", is_trainable=True)

lora_total, lora_trainable = count_parameters(model)
print(f"[LoRA Model] Total: {lora_total:,}, Trainable: {lora_trainable:,}")


from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_from_disk

models_dir = "./models/"
data_dir = "./data/datasets_cleaned/"
for ds_name in [
    "Amod-mental_health_counseling_conversations", "entfane-psychotherapy", "mrs83-kurtis_mental_health_final",
    "tcabanski-mental_health_counseling_responses", "ShenLab-MentalChat16K", "Psychotherapy-LLM-PsychoCounsel-Preference"
]:
    export_path = models_dir + ds_name + "/"
    
    ds_dct = load_from_disk(data_dir + ds_name)
    
    training_args = TrainingArguments(
        output_dir=export_path,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        learning_rate=5e-5,
        warmup_steps=512,
        logging_steps=512,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        optim="adamw_torch",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds_dct["train"],
        eval_dataset=ds_dct["test"],
        args=training_args,
    )
    trainer.train()
    
    torch.cuda.empty_cache()

    training_args = TrainingArguments(
        output_dir=export_path + "test",
        per_device_train_batch_size=1,
        num_train_epochs=2,
        learning_rate=3e-5,
        warmup_steps=512,
        logging_steps=512,
        save_strategy="epoch",
        optim="adamw_torch",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds_dct["test"],
        args=training_args,
    )
    trainer.train()

    model.save_pretrained(export_path)

    torch.cuda.empty_cache()

model.save_pretrained(models_dir)
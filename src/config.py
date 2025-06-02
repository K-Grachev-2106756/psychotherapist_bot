import os

from dotenv import load_dotenv


load_dotenv("1.env")


hf_token = os.getenv("HF_TOKEN")
base_model_name = os.getenv("BASE_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
cbt_lora_path = os.getenv("CBT_LORA_PATH")
emo_lora_path = os.getenv("CBT_EMO_LORA_PATH")
max_tokens = int(os.getenv("MAX_TOKENS", 2048))

class_mapping = {
    0: "admiration",
    1: "anger",
    2: "annoyance",
    3: "disappointment",
    4: "disapproval",
    5: "disgust",
    6: "excitement",
    7: "gratitude",
    8: "joy",
    9: "optimism",
    10: "sadness",
    11: "neutral",
}
clf_model_name = os.getenv("CLF_MODEL_NAME", "google-bert/bert-base-uncased")
clf_model_path = os.getenv("CLF_MODEL_PATH")

default_streamlit_model = "Base"
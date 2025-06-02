import torch
from transformers import (
    PreTrainedTokenizer,
    AutoModelForCausalLM,
)

from inference.llm import LLM
from inference.classifier import Classifier


class ChatHistory:

    def __init__(self, tokenizer: PreTrainedTokenizer, max_tokens: int = 2048):
        self.tokenizer = tokenizer

        self.fixed = []   # [(role, tokens)] — нельзя удалять
        self.history = [] # [(role, tokens)] — можно триммить

        self.max_tokens = max_tokens
        self.total_token_count = 0

        self.ask_tokens = self.tokenizer("THERAPIST:", return_tensors="pt", add_special_tokens=False)["input_ids"][0]


    def _tokenize(self, text: str, add_special_tokens: bool = False):
        return self.tokenizer(text + "\n", return_tensors="pt", add_special_tokens=add_special_tokens)["input_ids"][0]


    def set_system_prompt(self, text: str):
        tokens = self._tokenize(text.strip(), add_special_tokens=True)
        self.fixed.append(("system", tokens))
        self.total_token_count += len(tokens)


    def add_message(self, role: str, text: str, emotions: list = None):
        prefix = {
            "user": "PATIENT" + (" " + str(emotions) if emotions else "") + ": ",
            "assistant": "THERAPIST: ",
            "system": "",  # system уже обрабатывается отдельно
        }.get(role, "")

        tokens = self._tokenize(prefix + text.strip(), add_special_tokens=False)

        # Первое сообщение юзера → считаем фиксированным
        if (role == "user" and not any(r == "user" for r, _ in self.fixed)) or (role == "system"):
            self.fixed.append((role, tokens))
        else:
            self.history.append((role, tokens))
            self.trim_to_fit()
        self.total_token_count += len(tokens)


    def get_context(self, add_ask_tokens: bool = True):
        all_tokens = [t for _, t in self.fixed + self.history]
        if add_ask_tokens:
            all_tokens.append(self.ask_tokens)

        return torch.cat(all_tokens).unsqueeze(0)


    def trim_to_fit(self):
        # Удаляем только из истории
        while (self.total_token_count > self.max_tokens) or (len(self.history) > 4):
            _, tokens = self.history.pop(0)
            self.total_token_count -= len(tokens)


    def reset(self, keep_system_prompt: bool = True):
        if keep_system_prompt:
            self.history = []
            self.fixed = [(r, t) for r, t in self.fixed if r == "system"]
            self.total_token_count = sum(len(t) for _, t in self.fixed)
        else:
            self.history = []
            self.fixed = []
            self.total_token_count = 0


class Chat:

    def __init__(
            self, 
            llm_model: AutoModelForCausalLM,
            llm_tokenizer: PreTrainedTokenizer,
            classifier: Classifier = None,
            max_context_tokens: int = 2048,
    ):
        self.model, self.tokenizer = llm_model, llm_tokenizer
        self.classifier = classifier if classifier is not None else lambda _: None

        self.history = ChatHistory(self.tokenizer, max_tokens=max_context_tokens)
        self.history.set_system_prompt(
            "You are a licensed psychotherapist specializing in Cognitive Behavioral Therapy (CBT). "
            "Do not copy or quote any textbook or training material directly. Always answer in your own words, clearly and compassionately. "
            "Don't stop generating until you finish the sentence."
            "You only respond as the THERAPIST. Never write dialogue for the PATIENT. "
            "Only write one response from the THERAPIST per prompt. "
            "Ask questions that will help you solve the patient's problem. "
            "Do not ask too many questions. "
        )

        self.device = torch.device("cuda:0")


    def ask(self, user_input: str, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9) -> str:
        self.history.add_message("user", user_input, emotions=self.classifier(user_input))

        input_ids = self.history.get_context(add_ask_tokens=True).to(self.device)
        attention_mask = torch.ones(size=input_ids.shape).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = output_ids[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        answer = None  # Обработка ответа модели
        for generated_part in response.split("THERAPIST"):
            if generated_part:
                for therapist_ans in generated_part.split("PATIENT"):
                    if therapist_ans:
                        answer = therapist_ans.strip()
                        break
                break
        
        if answer is None: 
            answer = response.strip()
        
        for i, ch in enumerate(answer[::-1]):
            if (ch in {".", "!", "?"}) and (i < len(answer)):
                if i != 0:
                    answer = answer[:-i].strip()
                break
        
        self.history.add_message("assistant", answer)

        torch.cuda.empty_cache()
        
        return answer


    def reset(self):
        self.history.reset()




if __name__ == "__main__":
    base_llm, tuned_llm, llm_tokenizer = LLM.init(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        model_path="/mnt/sdb1/home/kygrachev/diploma/models/textbook/mistral-psy-lora",
    )

    cht = Chat(tuned_llm, llm_tokenizer)
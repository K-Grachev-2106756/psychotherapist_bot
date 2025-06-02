from typing import Union

import torch
import torch.nn as nn
from transformers import (
    BertModel,
    AutoTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput


class MultilabelFocalLoss:

    def __init__(self, pos_weight=None):
        self.pos_weight = pos_weight


    def __call__(self, logits, targets, alpha=0.25, gamma=2.0, eps=1e-5):
        targets = targets.float()

        if self.pos_weight is not None:
            alphas = ((targets * (self.pos_weight - 1)) + alpha * (1 - targets)) 
        else:
            alphas = alpha
      
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_loss = -alphas * ((1 - pt) ** gamma * torch.log(pt + eps))

        return focal_loss.sum()


class BertForMultiLabelClassification(nn.Module):

    def __init__(
            self, 
            model_name: str,
            num_labels: int, 
            device: torch.device, 
            pos_weight: torch.Tensor = None, 
        ):
        super(BertForMultiLabelClassification, self).__init__()

        self.bert = BertModel.from_pretrained(model_name).to(device)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels, device=device)
        self.threshold_levels = nn.Parameter(torch.zeros(num_labels, device=device) * 0.5, requires_grad=True)
        
        if isinstance(pos_weight, torch.Tensor):
            pos_weight = pos_weight.to(device)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.threshold_loss_fn = MultilabelFocalLoss(pos_weight=pos_weight)

        self.device = device
        
        self.eval()


    def forward(self, **inputs) -> SequenceClassifierOutput:
        labels = inputs.pop("labels") if "labels" in inputs else None

        outputs = self.bert(**inputs)
        pooled_output = outputs.pooler_output  # [CLS] токен
        x = self.dropout(pooled_output)
        logits = self.classifier(x)  # логиты
        
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss = self.loss_fn(logits, labels)  # labels.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
    @property
    def thresholds(self):
        return torch.sigmoid(self.threshold_levels)  # Ограничение порогов (0; 1)
    

class Classifier:

    def __init__(self, model_name: str, model_path: str, class_mapping: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_mapping = class_mapping

        self.model = BertForMultiLabelClassification(
            model_name=model_name,
            num_labels=len(self.class_mapping), 
            pos_weight=None, 
            device=self.device,
        )

        if model_path:
            self.model.load_state_dict(torch.load(model_path), strict=False)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def __call__(self, text: Union[str, list]):
        data = text if isinstance(text, list) else [text]

        inputs = self.tokenizer(
            data, 
            padding="max_length", 
            max_length=512, 
            truncation=True,
        )

        with torch.no_grad():
            outputs = self.model.forward(**{
                k: torch.tensor(inputs[k]).to(self.device)
                for k in ["input_ids", "attention_mask"]
            })
        preds = (torch.sigmoid(outputs.logits) > self.model.thresholds).int().tolist()
        
        result = []
        for pred in preds:
            response = [self.class_mapping[i] for i, v in enumerate(pred) if v == 1]
            if len(response) == 0:
                response.append("neutral")
            result.append(response)
        
        if isinstance(text, list):
            return result
        
        return result[0]




if __name__ == "__main__":
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

    clf = Classifier(
        model_name="google-bert/bert-base-uncased",
        model_path="./models/classifier/optimize_levels_on_test/model_best/model.pt",
        class_mapping=class_mapping,
    )

    print(clf("i'm so tired you know.. just go away"))
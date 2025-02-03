# triple2question.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Bert(nn.Module):
    def __init__(
        self, model_name="bert-base-chinese", device=None
    ):
        super(RandengT5, self).__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

        # 设置设备（默认自动选择）
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        self.model.to(self.device)

    def forward(self, input_ids, attention_mask, labels=None):
        # 不需要手动移动设备，因为模型已经在设备上
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            labels=labels.to(self.device) if labels is not None else None,
        )
        return outputs

    def generate(self, input_text, max_length=100):
        try:
            print(f"输入文本: {input_text}")
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(
                self.device
            )
            output_ids = self.model.generate(input_ids, max_length=max_length)
            generated_text = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            print("生成的问题为：", generated_text, output_ids)
            return generated_text
        except Exception as e:
            print(f"生成过程中发生错误: {e}")
            return None

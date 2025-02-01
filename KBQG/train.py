import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, T5Tokenizer
from tqdm import tqdm
from model.triple2question import Triple2QuestionModel
import os

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


# 自定义数据集
class TripleDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.data = json.load(open(data_path, "r", encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"generate question: {item['path']}"
        print(input_text)
        target_text = item["q"]

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }


# 训练函数
def train(model, dataloader, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            # 确保所有张量都在同一设备上
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader)}")


# 主函数
if __name__ == "__main__":
    # 初始化分词器并设置 legacy=False
    tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)

    # 加载数据
    dataset = TripleDataset("./data/train_converted.json", tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    # 加载模型并确保其在正确的设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Triple2QuestionModel().to(device)

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # 训练模型
    train(model, dataloader, optimizer, device, epochs=10)

    # 保存模型
    torch.save(model.state_dict(), "triple2question_model.pth")

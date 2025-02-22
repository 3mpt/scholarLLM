import sys
import os

# 获取 `KBQG` 目录的路径
KBQG_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 把 `KBQG` 目录添加到 Python 搜索路径
sys.path.append(KBQG_PATH)
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from model.triple2question import Triple2QuestionModel
from model.randeng_T5 import RandengT5
from model.bart import Bart
from datetime import datetime

# 设置代理
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
        input_text = f"给定知识：{item['path']}，问题的答案是：{item['answer']}，请生成问题"
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
def train(model, dataloader, val_dataloader, optimizer, device, epochs=5):
    start_time = datetime.now()
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_path = None  # 记录当前最优模型路径

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

        # 进入验证模式
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_input_ids = val_batch["input_ids"].to(device)
                val_attention_mask = val_batch["attention_mask"].to(device)
                val_labels = val_batch["labels"].to(device)
                val_outputs = model(
                    input_ids=val_input_ids, attention_mask=val_attention_mask, labels=val_labels
                )
                val_loss += val_outputs.loss.item()

        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader)} Val Loss: {val_loss / len(val_dataloader)}")

        # 只保留最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            # 生成新模型的路径
            current_time = datetime.now().strftime("%m_%d_%H_%M")
            model_save_path = f"../output/model/bart_ans.pth"

            # 如果之前保存了模型，则删除
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f"Removed previous best model: {best_model_path}")

            # 保存新模型
            torch.save(model.state_dict(), model_save_path)
            best_model_path = model_save_path
            print(f"Best model saved at epoch {epoch + 1} with Val Loss: {val_loss}")

    end_time = datetime.now()
    total_time = end_time - start_time
    print(f"Total Training Time: {total_time}")

# 主函数
if __name__ == "__main__":
    # 初始化分词器并设置 legacy=False
    tokenizer = AutoTokenizer.from_pretrained("fnlp/bart-base-chinese", legacy=False)

    # 加载数据
    train_dataset = TripleDataset("../data/ALL/train.json", tokenizer)
    val_dataset = TripleDataset("../data/ALL/val.json", tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 加载模型并确保其在正确的设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Triple2QuestionModel().to(device)
    # model = RandengT5().to(device)
    model = Bart().to(device)
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # 训练模型
    train(model, train_dataloader, val_dataloader, optimizer, device, epochs=5)

from transformers import AutoTokenizer, AutoModelForCausalLM

# 自定义模型下载路径
save_path = "../../models/Qwen2.5-7B-Instruct"

# 下载模型到指定路径
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", cache_dir=save_path)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", cache_dir=save_path)

print(f"模型已下载到: {save_path}")

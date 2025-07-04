from torch.utils.data import random_split, DataLoader
from pretrain_dataloader import ProductReviewDataset
from transformers import BertTokenizerFast
from transformers import BertConfig, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
import os
import torch

# 数据拆分比例
train_ratio = 0.9
test_ratio = 0.1
# 预设参数
vocab_size = 30522  # BERT模型的默认词表大小
max_length = 512    # 最大序列长度
model_path = "./pretrained-bert"  # 模型保存路径

# 加载保存的分词器
tokenizer = BertTokenizerFast.from_pretrained("./pretrained-bert")

def collate_fn(batch):
    # 从batch中提取句子
    sents = [item["sentence"] for item in batch]

    # 使用tokenizer的__call__方法，自动进行tokenization和padding
    data = tokenizer(
        sents,
        truncation=True,
        padding=True,           # 自动填充到最长句子长度
        max_length=max_length, # 最大序列长度
        return_tensors='pt'    # 返回PyTorch张量
    )

    # 提取编码后的input_ids和attention_mask
    input_ids = data["input_ids"]           # 句子的token ID
    attention_mask = data["attention_mask"] # 注意力掩码
    token_type_ids = data["token_type_ids"] # 获取token类型ID，如果没有则为None


    # 返回字典格式
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    }

# 创建数据集实例
txt_file = "./dataset/Amason/amazon_reviews.txt"
dataset = ProductReviewDataset(txt_file)

# 计算训练集和测试集的样本数量
dataset_size = len(dataset)
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size  # 剩余的部分作为测试集

# 使用 random_split 进行数据集拆分
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建 DataLoader 进行批量处理
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)



# **1. 初始化模型**
model_config = BertConfig(
    vocab_size=vocab_size,  # 词表大小
    max_position_embeddings=max_length  # 最大位置嵌入长度
)
model = BertForMaskedLM(config=model_config)  # 加载掩码语言模型 (MLM)

# **2. 数据掩码 (Masking)**
# 设置 Data Collator，随机 Mask 20% 的 token (默认为 15%)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.2
)

# **3. 训练参数**
training_args = TrainingArguments(
    output_dir=model_path,           # 输出模型的保存路径
    evaluation_strategy="steps",     # 每隔指定步数评估模型
    overwrite_output_dir=True,       # 如果目录存在，允许覆盖
    num_train_epochs=10,             # 训练轮次
    per_device_train_batch_size=10,  # 每个设备的训练批量大小
    gradient_accumulation_steps=8,   # 梯度累积步数
    per_device_eval_batch_size=64,   # 每个设备的评估批量大小
    logging_steps=1000,              # 每 1000 步记录日志
    save_steps=1000,                 # 每 1000 步保存模型
    save_total_limit=3,              # 最多保存 3 个模型检查点
)

# **4. 初始化 Trainer**
trainer = Trainer(
    model=model,                     # 训练的模型
    args=training_args,              # 训练参数
    data_collator=data_collator,     # 数据掩码方法
    train_dataset=train_dataset,     # 训练数据集
    eval_dataset=test_dataset        # 测试数据集
)

# **5. 训练模型**
trainer.train()



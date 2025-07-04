from transformers import BertTokenizerFast
from tokenizers import BertWordPieceTokenizer
import os

txt_file = "./dataset/Amason/all.txt"
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]

# 设定词表大小，BERT 模型 vocab size 为 30522，可自行更改
vocab_size = 30522

# 最大序列长度 (BERT 模型默认 512)
max_length = 512

# 截断处理标志，若文本长度超过最大长度则截断
truncate_longer_samples = True

# 初始化 WordPiece tokenizer
tokenizer = BertWordPieceTokenizer()

# 文件路径列表（需要确保此处路径正确）
files = [txt_file]

# 训练 tokenizer
tokenizer.train(
    files=files, 
    vocab_size=vocab_size, 
    special_tokens=special_tokens
)

# 截断至最大序列长度
if truncate_longer_samples:
    tokenizer.enable_truncation(max_length=max_length)

# 测试 tokenizer
test_sentence = "This product is really amazing and worth every penny!"
encoding = tokenizer.encode(test_sentence)

print("测试句子:", test_sentence)
print("Tokenized 输出:", encoding.tokens)
print("Token IDs:", encoding.ids)
print("Attention Mask:", encoding.attention_mask)

model_path = "pretrained-bert"
# make the directory if not already there
if not os.path.isdir(model_path):
    os.mkdir(model_path)

# 存储tokenizer（保存词表和合并文件）
tokenizer.save_model(model_path)

# 现在无需手动创建 config.json，BertTokenizerFast 会自动处理
# 加载已经训练好的 BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained(model_path)

# 测试加载后的 tokenizer
encoding = tokenizer(test_sentence)

print("加载后的 Tokenized 输出:", encoding.tokens)
print("加载后的 Token IDs:", encoding.input_ids)
print("加载后的 Attention Mask:", encoding.attention_mask)
print("加载后的 Attention Mask:", encoding.token_type_ids)

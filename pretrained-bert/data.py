import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
nlp = spacy.load("en_core_web_sm")
# 读取数据集
df = pd.read_csv("./dataset/Amason/amazon_reviews.txt", sep="\t")

# 提取 REVIEW_TEXT 列
reviews = df['REVIEW_TEXT']

# 按照 9:1 划分训练集和测试集
train_reviews, test_reviews = train_test_split(reviews, test_size=0.1, random_state=42)

# 定义函数：提取句法依赖关系
def extract_dependencies(texts, output_path):
    """
    提取每个句子的句法依赖关系。
    每一行包含：单词、词性标签、依赖关系和修饰词的头。
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in nlp.pipe(texts, batch_size=50):
            for token in doc:
                # 写入单词、词性、依赖关系及其头
                f.write(f" Node: {token.text} Part of Speech: {token.pos_} Dependencies: {token.dep_} Head: {token.head.text}")
            f.write("\n")  # 用空行分隔句子
# 提取训练集和测试集的依赖关系
extract_dependencies(train_reviews, "./dataset/Amason/train_dependencies.txt")
extract_dependencies(test_reviews, "./dataset/Amason/test_dependencies.txt")
extract_dependencies(reviews, "./dataset/Amason/all.txt")

# 打印数据集的大小
print(f"训练集大小: {len(train_reviews)}")
print(f"测试集大小: {len(test_reviews)}")

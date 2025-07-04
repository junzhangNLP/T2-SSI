import pandas as pd
import torch
from torch.utils.data import Dataset
import spacy
import numpy as np

# 加载 spaCy 英语模型
nlp = spacy.load("en_core_web_sm")

class ProductReviewDataset(Dataset):
    def __init__(self, txt_file):
        """
        初始化数据集
        :param txt_file: 包含数据的 TXT 文件路径
        :param dic_path: 包含词向量和市场情绪的 JSON 文件路径
        """
        # 读取制表符分隔的 TXT 文件
        self.data = pd.read_csv(txt_file, sep='\t', quoting=3)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 读取当前索引的数据
        item = self.data.iloc[idx]
        sentence = item['REVIEW_TEXT']

        # 返回结果字典
        return {
            'sentence': sentence
        }


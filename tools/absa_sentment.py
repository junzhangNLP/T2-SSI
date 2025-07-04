import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
import spacy
import numpy as np
nlp = spacy.load("en_core_web_sm")

class SemevalDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        id_ = item['_id']
        sentence = item['sentence']
        sentiment = item['score']
        type_ = item['target']
        doc = nlp(sentence)
        for token in doc:
            G_input = f" Node: {token.text} {token.pos_} {token.dep_} {token.head.text}"
        sample = {
            'id': torch.tensor(id_, dtype=torch.long),
            'sentence': sentence, 
            'sentiment': torch.tensor(sentiment, dtype=torch.float),
            'target': type_,
            'G_input':G_input
        }
        return sample
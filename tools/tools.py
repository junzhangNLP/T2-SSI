import numpy as np
from sklearn.metrics import accuracy_score, f1_score,mean_squared_error, r2_score,precision_score,recall_score
import torch
import torch.nn as nn
import pynvml
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import random

def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def assign_gpu(gpu_ids, memory_limit=1e16):
    if len(gpu_ids) == 0 and torch.cuda.is_available():
        pynvml.nvmlInit()
        n_gpus = pynvml.nvmlDeviceGetCount()
        dst_gpu_id, min_mem_used = 0, memory_limit
        for g_id in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        gpu_ids.append(dst_gpu_id)

    using_cuda = len(gpu_ids) > 0 and torch.cuda.is_available()

    device = torch.device('cuda:%d' % int(gpu_ids[0]) if using_cuda else 'cpu')
    return device

def cosine_similarity(G, P):
    dot_product = np.dot(G, P)
    norm_G = np.linalg.norm(G)
    norm_P = np.linalg.norm(P)
    if norm_G == 0 or norm_P == 0:
        return 0.0
    return dot_product / (norm_G * norm_P)
    
def categorize_predictions(preds):
    """将预测值映射为三分类标签：-1, 0, 1"""
    categorized_preds = np.where(preds < -0.3, -1, np.where(preds > 0.3, 1, 0))
    return categorized_preds
    
def calculate_absa_metrics(results, truths):
    # 将预测结果和真实值从张量转换为 numpy 数组
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()
    
    # 将预测值映射为三分类标签
    test_preds_categorized = categorize_predictions(test_preds)

    # 确保预测和真实值的形状适合余弦相似度计算
    test_preds_2d = test_preds.reshape(-1, 1)
    test_truth_2d = test_truth.reshape(-1, 1)

    # 计算精确度、召回率和 F1 分数，适用于三分类
    accuracy = accuracy_score(test_truth, test_preds_categorized)
    precision = precision_score(test_truth, test_preds_categorized, average='macro')  # 或 'weighted'
    recall = recall_score(test_truth, test_preds_categorized, average='macro')  # 或 'weighted'
    f_score = f1_score(test_truth, test_preds_categorized, average='macro')  # 或 'weighted'
    
    # 计算余弦相似度
    cosine_sim = cosine_similarity(test_truth, test_preds)
    
    # 计算加权系数
    weight_cosine = len(test_preds) / len(test_truth)
    
    # 计算最终的余弦相似度得分
    final_cosine_score = weight_cosine * cosine_sim
    
    # 计算 MSE
    mse = mean_squared_error(test_truth, test_preds)
    
    # 计算 R²
    ss_res = np.sum((test_preds - test_truth) ** 2)
    ss_tot = np.sum((test_truth - np.mean(test_truth)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    conf_matrix = confusion_matrix(test_truth, test_preds_categorized)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f_score,
        'final_cosine_score': final_cosine_score,
        'mse': mse,
        'r2': r2
    }
    print("Evaluation Metrics:")
    print("===================")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1 Score:           {metrics['f1_score']:.4f}")
    print(f"Final Cosine Score: {metrics['final_cosine_score']:.4f}")
    print(f"MSE:                {metrics['mse']:.4f}")
    print(f"R^2:                {metrics['r2']:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    return metrics

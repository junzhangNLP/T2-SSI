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

def calculate_metrics(results, truths):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()
    binary_truth_has0 = test_truth >= 0
    binary_preds_has0 = test_preds >= 0
    # 计算精确度、召回率和 F1 分数
    accuracy = accuracy_score(binary_truth_has0, binary_preds_has0)
    precision = precision_score(binary_truth_has0, binary_preds_has0)
    recall = recall_score(binary_truth_has0, binary_preds_has0)
    f_score = f1_score(binary_truth_has0, binary_preds_has0, average='weighted')



    metrics = {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1_score":f_score,
    }
    print("Evaluation Metrics:")
    print("===================")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1 Score:           {metrics['f1_score']:.4f}")
    return metrics
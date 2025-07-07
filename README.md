# T2-SSI
Previous graph-based approaches in Aspect-Based Sentiment Analysis (ABSA) have demonstrated impressive performance by modeling syntactic dependencies through graph neural networks (GNNs) and capturing semantic context through Transformers. However, to fuse features from different models, these approaches have to sacrifice fine-grained information in the fusion stage, potentially resulting in the mutual influence of different aspect words, thus leading to incorrect predictions. To address this issue, we propose a Dual-Transformer-Based Fine-grained Fusion Network for Syntax-Semantics Interactive in Aspect-Based Sentiment Analysis (T$^2$-SSI). Specifically, to balance the differences between the syntactic encoder and semantic encoder in the process of fine-grained feature extraction, we integrate prompt learning and node prediction tasks to train a Transformer-based syntactic feature encoder. We further propose the Fine-grained Cross-fusion Network with Cross-Fusion Attention to improve the information interaction between syntactic and semantic at the fine-grained level. The network realizes the fine-grained interaction among aspect words, syntactic, and semantic information to decrease the mutual influence between aspect words, thus achieving more effective ABSA. Experimental results on three benchmarks demonstrate the effectiveness of our framework.

# data
Datasets and pre-training data can be downloaded from the network disk. 
（link: https://pan.baidu.com/s/1GhZtDcdDB-aiNMlLMLiDgA?pwd=aaai code: aaai）

#pretrain
A trained syntactic encoder is provided in the network disk. If you want to train a syntax encoder based on new content, please refer to the following steps.
1.Put data into the data folder.
2.Modify the path in the pre-training code.
3.Use the code in the ./pretrained-bert folder to generate a tokenizer for the new content.
4.The pre-training process refers to this repository: https://github.com/circlePi/Pretraining-Yourself-Bert-From-Scratch

#train
You should first check whether the file path is correct. Download the universal Bert model (bert-base-uncased) through huggingface. Start train_rest.py to run.


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch_geometric.nn import GCNConv
import math
torch.autograd.set_detect_anomaly(True)
# GCN 模型定义
from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageEmbeddingLayer(nn.Module):
    def __init__(self):
        super(LanguageEmbeddingLayer, self).__init__()
        self.bertmodel = AutoModel.from_pretrained('./bert-base-uncased')

    def forward(self, bert_sent, bert_sent_mask):
        output = self.bertmodel(input_ids=bert_sent, attention_mask=bert_sent_mask)
        text = output[0]
        return text
    
class GEmbeddingLayer(nn.Module):
    def __init__(self):
        super(GEmbeddingLayer, self).__init__()
        self.bertmodel = AutoModel.from_pretrained("./encoder/bert")

    def forward(self, bert_sent, bert_sent_mask):
        output = self.bertmodel(input_ids=bert_sent, attention_mask=bert_sent_mask)
        text = output[0]
        return text
    
class CrossModalAttention(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(CrossModalAttention, self).__init__()
        self.activation = nn.ReLU()
        self.x_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.y_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.t_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.x_weight_1.data.fill_(1)
        self.y_weight_1.data.fill_(1)
        self.t_weight_1.data.fill_(1)
        self.bias.data.fill_(0)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self,  x, y, target):
        x_att = torch.matmul(x, x.transpose(-1, -2))
        x_att = self.activation(x_att)

        y_att = torch.matmul(y, y.transpose(-1, -2))
        y_att = self.activation(y_att)

        t_att = torch.matmul(target, target.transpose(-1, -2))
        t_att = self.activation(t_att)

        fusion_att = self.x_weight_1 * x_att + self.y_weight_1 * y_att + self.t_weight_1 * t_att + self.bias
        fusion_att = nn.Softmax(dim=-1)(fusion_att)
        target_att = torch.matmul(fusion_att, target)

        dropped = self.dropout(target_att)
        y_1 = F.relu(self.linear_1(dropped), inplace=True)
        y_2 = F.relu(self.linear_2(y_1), inplace=True)
        return y_2
    
class FusionSubNet(nn.Module):

    def __init__(self, in_size):
        super(FusionSubNet, self).__init__()
        self.rnn = nn.GRU(in_size, in_size, num_layers=1, dropout=0, bidirectional=True, batch_first=True)
        self.linear_1 = nn.Linear(2 * in_size, in_size)
        self.linear_2 = nn.Linear(in_size, in_size)

    def forward(self, h, p):
        output, _ = self.rnn(h)
        # print(output.shape)
        a_1 = F.relu(self.linear_1(output), inplace=True)
        a_2 = nn.Sigmoid()(self.linear_2(a_1))
        y = torch.matmul(a_2.permute(1, 0), p).squeeze()
        return y,a_1
    
class LF_DNN(nn.Module):

    def __init__(self, text_in, text_hidden, G_in, G_hidden, text_prob, post_text_prob, post_text_dim):
        super(LF_DNN, self).__init__()
        self.text_in = text_in
        self.text_hidden = text_hidden
        self.G_in = G_in
        self.G_hidden = G_hidden
        self.text_prob = text_prob
        self.post_text_prob = post_text_prob
        self.post_text_dim = post_text_dim
        self.output_dim = 1

        self.text_enc_bert = LanguageEmbeddingLayer()
        self.tar_enc_bert = LanguageEmbeddingLayer()
        self.G_enc_bert = GEmbeddingLayer()

        self.attention_layer_1 = nn.Linear(self.text_in, self.text_hidden)
        self.attention_layer_2 = nn.Linear(self.G_in, self.G_hidden)
        self.attention_layer_3 = nn.Linear(self.text_in, self.text_hidden)
        self.text_cutnet = CrossModalAttention(self.text_hidden, self.text_prob)
        self.G_cutnet = CrossModalAttention(self.G_hidden, self.text_prob)
        self.target_cutnet = CrossModalAttention(self.G_hidden, self.text_prob)
        self.post_fusion_dropout = nn.Dropout(p=self.text_prob)

        self.post_fusion_layer_1 = nn.Linear(2 * self.text_hidden, self.text_hidden)
        self.post_fusion_layer_2 = nn.Linear(self.text_hidden, self.post_text_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_text_dim, self.output_dim)
        self.fusion_subnet = FusionSubNet(self.text_hidden)
        self.fusion_layer_1 = nn.Linear(256, self.post_text_dim)
        self.fusion_layer_2 = nn.Linear(self.post_text_dim, self.output_dim)

    def forward(self, input_ids_tar, att_mask_tar, G_input_ids, G_attention_mask,input_ids, att_mask):
        text_cutx = self.text_enc_bert(input_ids, att_mask)   
        G_cutx = self.G_enc_bert(G_input_ids, G_attention_mask)  
        tar_cutx = self.tar_enc_bert(input_ids_tar,att_mask_tar) 
        fusion_batchs = list()
        for i in range(len(text_cutx)):
            text_cutxi = text_cutx[i]
            tar_cutxi = tar_cutx[i]
            tar_cutxi = tar_cutxi[0, :].repeat(50, 1)
            G_cutxi = G_cutx[i]

            text_cutxi = self.attention_layer_1(text_cutxi)
            G_cutxi = self.attention_layer_2(G_cutxi)

            text_cuth = self.text_cutnet(tar_cutxi, G_cutxi,text_cutxi)
            G_cuth = self.G_cutnet(text_cutxi, tar_cutxi,G_cutxi)

            fusion_h = torch.cat([G_cuth, text_cuth], dim=-1)
            dropped = self.post_fusion_dropout(fusion_h)
            x_1 = F.relu(self.post_fusion_layer_1(dropped), inplace=True)
            x_2 = F.relu(self.post_fusion_layer_2(x_1), inplace=True)
            _fusion = self.post_fusion_layer_3(x_2)
            fusion,_ = self.fusion_subnet(x_1, _fusion)

            fusion_batchs.append(fusion.unsqueeze(0))
        fusions = torch.cat(fusion_batchs, dim=0)
        f1 = self.fusion_layer_1(fusions)
        output_fusion = self.fusion_layer_2(f1)
        res = {
            "pre_t":output_fusion,
        }
        return res

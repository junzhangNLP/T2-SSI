from tools.absa_sentment import SemevalDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, BertTokenizerFast
from tools.tools import assign_gpu,setup_seed,calculate_absa_metrics
from model.GBGS import LF_DNN
from torch import optim
from tqdm import tqdm
datalist = ["twitter","rest14","lap14"]
# 加载数据集
sem_test = f'./dataset/{datalist[0]}/test.csv'  
sem_dev = f'./dataset/{datalist[0]}/dev.csv'  
sem_train = f'./dataset/{datalist[0]}/train.csv' 

dataset_test = SemevalDataset(sem_test)
dataset_dev = SemevalDataset(sem_dev)
dataset_train = SemevalDataset(sem_train)

# 创建数据加载器
batch_size = 16
max_length = 50
indices = list(range(len(dataset_train)))

model_path = "./bert-base-uncased"
G_model_path = "./encoder/bert"
sentence_tokenizer = RobertaTokenizer.from_pretrained(model_path)
G_tokenizer = BertTokenizerFast.from_pretrained(G_model_path)

def collate_fn(batch):
    sents = [i["sentence"].replace(i["target"],"$T$") for i in batch]
    target = [i["sentiment"] for i in batch]
    tars = [i["target"] for i in batch]
    Gfeatures = [i["G_input"] for i in batch]

    G_data = G_tokenizer.batch_encode_plus(
        batch_text_or_text_pairs = Gfeatures,
        truncation = True,
        padding='max_length',
        max_length=max_length,
        return_length=True,
        return_tensors='pt'
    )

    data = sentence_tokenizer.batch_encode_plus(
        batch_text_or_text_pairs = sents,
        truncation = True,
        padding='max_length',
        max_length=max_length,
        return_length=True,
        return_tensors='pt'
    )

    tars = sentence_tokenizer.batch_encode_plus(
        batch_text_or_text_pairs = tars,
        truncation = True,
        padding='max_length',
        max_length=5,
        return_length=True,
        return_tensors='pt'
    )
    input_ids = data['input_ids']
    att_mask = data["attention_mask"]
    input_ids_combined = []
    attention_mask_combined = []

    for target_ids, sent_ids in zip(tars['input_ids'], data['input_ids']):
        combined_ids = torch.cat([target_ids[:-1], sent_ids[1:]], dim=0) 
        attention_mask = (combined_ids != sentence_tokenizer.pad_token_id).long()  
        
        if combined_ids.size(0) > max_length:
            combined_ids = combined_ids[:max_length] 
            attention_mask = attention_mask[:max_length] 
        else:
            padding = torch.zeros(max_length - combined_ids.size(0), dtype=torch.long)
            combined_ids = torch.cat([combined_ids, padding], dim=0)
            attention_mask = torch.cat([attention_mask, padding], dim=0)
        
        input_ids_combined.append(combined_ids)
        attention_mask_combined.append(attention_mask)

    input_ids_tar = torch.stack(input_ids_combined)
    att_mask_tar = torch.stack(attention_mask_combined)

    target = torch.stack(target)

    G_input_ids = G_data['input_ids']
    G_attention_mask = G_data['attention_mask']
    return input_ids_tar, att_mask_tar, target, G_input_ids, G_attention_mask, input_ids, att_mask

test_data_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dev_data_loader = DataLoader(dataset_dev, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
train_data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

best_valid = 1e8
setup_seed(1111)
device = assign_gpu([0])
train_mode = "regression"

text_in = 768
text_hidden = 256
G_in = 768
G_hidden = 256

text_prob = 0.1
post_text_prob  = 0.1

post_text_dim = 16

criterion = nn.MSELoss()
model = LF_DNN(text_in, text_hidden, G_in, G_hidden, text_prob, post_text_prob, post_text_dim)

model_params_other = [p for n, p in list(model.named_parameters()) if 'text_enc_bert' not in n and 'G_enc_bert' not in n and 'tar_enc_bert' not in n]

optimizer = optim.AdamW(
    [
     {"params": list(model.text_enc_bert.parameters()),"lr":1e-5, "weight_decay": 1e-4},
     {"params": list(model.G_enc_bert.parameters()),"lr":1e-5, "weight_decay": 1e-4},
     {"params": list(model.tar_enc_bert.parameters()),"lr":1e-5, "weight_decay": 1e-4},
     {'params': model_params_other}],
    lr=0.0002, weight_decay=0.001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

def do_test(model, dataloader,criterion):
    model.eval()
    y_pred = []
    y_true = []
    eval_loss = 0.0
    with torch.no_grad():
        for datas in tqdm(dataloader):
            input_ids_tar, att_mask_tar, target, G_input_ids, G_attention_mask,input_ids, att_mask = datas[0].cuda(), datas[1].cuda(), datas[2].cuda().float(), datas[3].cuda() ,datas[4].cuda(), datas[5].cuda() ,datas[6].cuda()

            outputs = model(input_ids_tar, att_mask_tar, G_input_ids, G_attention_mask,input_ids, att_mask)

            loss = 0.0
            m_loss = criterion(outputs["pre_t"], target.unsqueeze(1)) 

            loss +=  m_loss 

            eval_loss += loss.item()
                    
            y_pred.append(outputs["pre_t"].detach().cpu())
            y_true.append(target.detach().cpu())

    eval_loss = round(eval_loss / len(dataloader), 4)
    eval_results = {}
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = calculate_metrics(pred, true)

    eval_results = results
    eval_results['Loss'] = round(eval_loss, 4)
    return eval_results

epochs, best_epoch = 0, 0
model.to(device)
results_train = []

while True:
    epochs += 1
    y_pred = []
    y_true = []
    train_loss = 0.0
    model.train()
    for datas in tqdm(train_data_loader):
        input_ids_tar, att_mask_tar, target, G_input_ids, G_attention_mask,input_ids, att_mask = datas[0].cuda(), datas[1].cuda(), datas[2].cuda().float(), datas[3].cuda() ,datas[4].cuda(), datas[5].cuda() ,datas[6].cuda()

        optimizer.zero_grad()
        loss = 0.0
        outputs = model(input_ids_tar, att_mask_tar, G_input_ids, G_attention_mask,input_ids, att_mask)

        m_loss =  criterion(outputs["pre_t"], target.unsqueeze(1))
        loss = m_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        y_pred.append(outputs["pre_t"].cpu())
        y_true.append(target.cpu())
    train_loss = train_loss / len(dataset_train)
    print("epoch",epochs,"trainloss:",train_loss)
    pred, true = torch.cat(y_pred), torch.cat(y_true)

    dev_results = do_test(model, dev_data_loader, criterion)
    cur_valid = dev_results["Loss"]

    val_results = do_test(model, test_data_loader, criterion)
    cur_valid = val_results["Loss"]

    isBetter = cur_valid <= (best_valid - 1e-6)

    if isBetter:
        best_valid, best_epoch = cur_valid, epochs
    if epochs == 30:
        break
    

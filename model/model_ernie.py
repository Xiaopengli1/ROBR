import numpy as np
import pandas as pd
import csv
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer,BertConfig,AdamW,BertModel,AutoTokenizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Union, List


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', texts: List[str] = None,  label: Union[int, float] = 0):
        """
        Creates one InputExample with the given texts, guid and label

        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))
def read_samples(df):
    samples = []
    label = []
    for i in tqdm(range(len(df))):
        origin_q = df.loc[i]["query"]
        woman_q = df.loc[i]["woman_query"]
        man_q = df.loc[i]["man_query"]
        student_q = df.loc[i]["student_query"]
        old_q = df.loc[i]["old_query"]
        value = df.loc[i]["label"]
        doc = df.loc[i]["doc"]
        value = int(value)
        samples.append(InputExample(texts=[[origin_q, doc],[woman_q,doc],[man_q,doc],[student_q,doc],[old_q,doc]], label=value))
    return samples

def smart_batching_collate(batch):
    text0 = []
    text1 = []
    text2 = []
    text3 = []
    text4 = []
    
    labels = []

    for example in batch:
        # for idx, text in enumerate():
        #     print(idx,len(text))
        text0.append(example.texts[0])
        text1.append(example.texts[1])
        text2.append(example.texts[2])
        text3.append(example.texts[3])
        text4.append(example.texts[4])
        labels.append(example.label)
    tokenizer = AutoTokenizer.from_pretrained("model_path/Pre_trained_model/cn/ernie-3.0-base-zh")
    
    tokenized0 = tokenizer(text0, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
    tokenized1 = tokenizer(text1, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
    tokenized2 = tokenizer(text2, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
    tokenized3 = tokenizer(text3, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
    tokenized4 = tokenizer(text4, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
    labels = torch.tensor(labels, dtype=torch.long)

    # for name in tokenized0:
    #     tokenized0[name] = tokenized0[name].to(self._target_device)

    return tokenized0,tokenized1,tokenized2,tokenized3,tokenized4,labels


class Adapter(nn.Module):
    """
    The adapters first project the original
    d-dimensional features into a smaller dimension, m, apply
    a nonlinearity, then project back to d dimensions.
    """
    def __init__(self, size = 32, model_dim = 768):
        super().__init__()
        self.adapter_block = nn.Sequential(
            nn.Linear(model_dim, size),
            nn.ReLU(),
            nn.Linear(size, model_dim)
        )

    def forward(self, x):

        ff_out = self.adapter_block(x)
        # Skip connection
        adapter_out = ff_out + x

        return adapter_out
    
class BertClassificationModel(nn.Module):
    def __init__(self,model_name):
        super(BertClassificationModel, self).__init__()
        #加载预训练模型
        self.bert = BertModel.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        #定义线性函数
        self.adapter_origin = Adapter()
        self.adapter_woman = Adapter()
        self.adapter_man = Adapter()
        self.adapter_student = Adapter()
        self.adapter_old = Adapter()
        self.adapter_share = Adapter()

        self.gate_origin = nn.Linear(768, 2)
        self.gate_woman = nn.Linear(768, 2)
        self.gate_man = nn.Linear(768, 2)
        self.gate_student = nn.Linear(768, 2)
        self.gate_old = nn.Linear(768, 2)

        self.classfication = nn.Linear(768, 5)


    def forward(self, d0, d1, d2, d3, d4):
        #得到bert_output
        input_ids0,token_type_ids0,attention_mask0 = d0["input_ids"],d0["token_type_ids"],d0["attention_mask"]
        input_ids1,token_type_ids1,attention_mask1 = d1["input_ids"],d1["token_type_ids"],d1["attention_mask"]
        input_ids2,token_type_ids2,attention_mask2 = d2["input_ids"],d2["token_type_ids"],d2["attention_mask"]
        input_ids3,token_type_ids3,attention_mask3 = d3["input_ids"],d3["token_type_ids"],d3["attention_mask"]
        input_ids4,token_type_ids4,attention_mask4 = d4["input_ids"],d4["token_type_ids"],d4["attention_mask"]

        bert_output0 = self.bert(input_ids=input_ids0,token_type_ids=token_type_ids0, attention_mask=attention_mask0)["pooler_output"] # B * 768
        bert_output1 = self.bert(input_ids=input_ids1,token_type_ids=token_type_ids1, attention_mask=attention_mask1)["pooler_output"] # B * 768
        bert_output2 = self.bert(input_ids=input_ids2,token_type_ids=token_type_ids2, attention_mask=attention_mask2)["pooler_output"] # B * 768
        bert_output3 = self.bert(input_ids=input_ids3,token_type_ids=token_type_ids3, attention_mask=attention_mask3)["pooler_output"] # B * 768
        bert_output4 = self.bert(input_ids=input_ids4,token_type_ids=token_type_ids4, attention_mask=attention_mask4)["pooler_output"] # B * 768

        adapter_output0 = self.adapter_origin(bert_output0) # B * 768
        adapter_output1 = self.adapter_woman(bert_output1) # B * 768
        adapter_output2 = self.adapter_man(bert_output2) # B * 768
        adapter_output3 = self.adapter_student(bert_output3) # B * 768
        adapter_output4 = self.adapter_old(bert_output4) # B * 768

        adapter_share0 = self.adapter_share(bert_output0) # B * 768
        adapter_share1 = self.adapter_share(bert_output1) # B * 768
        adapter_share2 = self.adapter_share(bert_output2) # B * 768
        adapter_share3 = self.adapter_share(bert_output3) # B * 768
        adapter_share4 = self.adapter_share(bert_output4) # B * 768

        gate0 = self.gate_origin(bert_output0) # B * 2
        gate1 = self.gate_woman(bert_output1) # B * 2
        gate2 = self.gate_man(bert_output2) # B * 2
        gate3 = self.gate_student(bert_output3) # B * 2
        gate4 = self.gate_old(bert_output4) # B * 2


        # print(adapter_output1.shape)
        # print(gate1[:,0].unsqueeze(1).shape)
        output0 = adapter_output0 * gate0[:,0].unsqueeze(1) + adapter_share0+ gate0[:,1].unsqueeze(1) # B * 768
        output1 = adapter_output1 * gate1[:,0].unsqueeze(1) + adapter_share1+ gate1[:,1].unsqueeze(1) # B * 768
        output2 = adapter_output2 * gate2[:,0].unsqueeze(1) + adapter_share2+ gate2[:,1].unsqueeze(1) # B * 768
        output3 = adapter_output3 * gate3[:,0].unsqueeze(1) + adapter_share3+ gate3[:,1].unsqueeze(1) # B * 768
        output4 = adapter_output4 * gate4[:,0].unsqueeze(1) + adapter_share4+ gate4[:,1].unsqueeze(1) # B * 768

        output0_class = self.classfication(output0)
        output1_class = self.classfication(output1)
        output2_class = self.classfication(output2)
        output3_class = self.classfication(output3)
        output4_class = self.classfication(output4)

        return  output0_class, output1_class, output2_class, output3_class, output4_class
class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p, q):
        from torch.functional import F
        p = F.softmax(p, dim=1)
        q= F.softmax(q, dim=1)
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(p.log(), m) + self.kl(q.log(), m))
class robust_acc_loss(nn.Module):
    def __init__(self,alpha=0.5,beta = 5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.jsd = JSD()
        self.accuracy = nn.CrossEntropyLoss()
        
    def forward(self, label,o0,o1,o2,o3,o4):
        accuracy1 = self.accuracy(o0,label)
        accuracy2 = self.accuracy(o1,label)
        accuracy3 = self.accuracy(o2,label)
        accuracy4 = self.accuracy(o3,label)
        accuracy5 = self.accuracy(o4,label)
        accuracy = accuracy1+accuracy2+accuracy3+accuracy4+accuracy5
        robust1 = self.jsd(o0,o1)
        robust2 = self.jsd(o0,o2)
        robust3 = self.jsd(o0,o3)
        robust4 = self.jsd(o0,o4)
        robust5 = self.jsd(o1,o2)
        robust6 = self.jsd(o1,o3)
        robust7 = self.jsd(o1,o4)
        robust8 = self.jsd(o2,o3)
        robust9 = self.jsd(o2,o4)
        robust10 = self.jsd(o3,o4)
        robust = robust1+robust2+robust3+robust4+robust5+robust6+robust7+robust8+robust9+robust10

        return self.alpha * accuracy+self.beta*robust,self.alpha * accuracy,self.beta*robust
samples = pd.read_csv("data/industrial/industrial_full.csv")
samples.columns = ["query", "doc" ,"label", "woman_query", "man_query", "student_query", "old_query"]


train_dataloader = DataLoader(read_samples(samples[:250000]), shuffle=True, batch_size=600)
# val_dataloader = DataLoader(read_samples(samples)[350000:-40000], shuffle=True, batch_size=30)
# test_dataloader = DataLoader(read_samples(samples)[-40000:], shuffle=True, batch_size=30)
train_dataloader.collate_fn = smart_batching_collate

model = BertClassificationModel("model_path/Pre_trained_model/cn/ernie-3.0-base-zh")
model.to("cuda:0")
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
weight_decay = 0.01
optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
total_epochs = 2
lparm = {"lr":2e-5}
optimizer = torch.optim.AdamW(optimizer_grouped_parameters,**lparm)
loss = robust_acc_loss()
for epoch in range(total_epochs):
    for step, (d0,d1,d2,d3,d4,labels) in tqdm(enumerate(train_dataloader)):
        d0.to("cuda:0")
        d1.to("cuda:0")
        d2.to("cuda:0")
        d3.to("cuda:0")
        d4.to("cuda:0")
        labels = labels.to("cuda:0")

        optimizer.zero_grad()

        output0_class, output1_class, output2_class, output3_class, output4_class = model(d0,d1,d2,d3,d4)
        
        all_loss,accuracy_loss,robust_loss=loss(labels,output0_class, output1_class, output2_class, output3_class, output4_class)
        
        all_loss.backward()
        optimizer.step()

        if (step + 1) % 2 == 0:
            print("Train Epoch[{}/{}],step[{}/{}],loss:{:.6f},acc_loss:{:.6f},robust_loss:{:.6f}".format(epoch + 1, total_epochs, step + 1, len(train_dataloader),all_loss.item(),accuracy_loss.item(),robust_loss.item()))
        
        
    path = './span_bert_hide_model2.pkl'
    torch.save(model, path)# if (step + 1) % 50 == 0:
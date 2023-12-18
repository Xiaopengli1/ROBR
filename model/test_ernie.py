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
from sklearn.metrics import ndcg_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from tqdm import tqdm

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
        doc = df.loc[i]["content"]
        value = int(value)
        samples.append(InputExample(texts=[[origin_q, doc],[woman_q,doc],[man_q,doc],[student_q,doc],[old_q,doc]], label=value))
    return samples

samples = pd.read_csv("data/industrial/industrial_full.csv")
samples.columns = ["query", "doc" ,"label", "woman_query", "man_query", "student_query", "old_query"]


test_samples = samples[-30000:]
test_samples.reset_index(drop=True,inplace=True)
# train_dataloader = DataLoader(read_samples(samples[:350000]), shuffle=True, batch_size=600)
# val_dataloader = DataLoader(read_samples(samples)[350000:-40000], shuffle=True, batch_size=30)
# test_dataloader = DataLoader(read_samples(test_samples), shuffle=False, batch_size=600)
# test_dataloader.collate_fn = smart_batching_collate

ndcg_origin_5_list = []
ndcg_woman_5_list = []
ndcg_man_5_list = []
ndcg_student_5_list = []
ndcg_old_5_list = []

ndcg_origin_10_list = []
ndcg_woman_10_list = []
ndcg_man_10_list = []
ndcg_student_10_list = []
ndcg_old_10_list = []

map_origin_list = []
map_woman_list = []
map_man_list = []
map_student_list = []
map_old_list = []

# model = BertClassificationModel("download")
f = open('span_ernie_hide_model.pkl','rb')
model = torch.load(f,map_location='cuda:0')#可使用cpu或gpu
query_unique = list(test_samples["query"].unique())

model = model.to("cuda:0")
# all_data = []
for i in tqdm(range(len(query_unique))):
    sample_this_query = test_samples[test_samples["query"] == query_unique[i]]
    sample_this_query.reset_index(inplace=True,drop=True)
    if len(sample_this_query)<10:
        print("<10,skip")
        continue
    q_l = []
    qw_l = []
    qm_l = []
    qs_l = []
    qo_l = []

    labels = list(sample_this_query["label"])

    for item in range(len(sample_this_query)):
        # print(sample_this_query.loc[item])
        q = sample_this_query.loc[item]["query"]
        qw = sample_this_query.loc[item]["woman_query"]
        qm = sample_this_query.loc[item]["man_query"]
        qs = sample_this_query.loc[item]["student_query"]
        qo = sample_this_query.loc[item]["old_query"]
        c = sample_this_query.loc[item]["doc"]

        
        q_l.append([q,c])
        qw_l.append([qw,c])
        qm_l.append([qm,c])
        qs_l.append([qs,c])
        qo_l.append([qo,c])
    

    tokenizer = AutoTokenizer.from_pretrained("model_path/Raw_model/cn/ernie-3.0-base-zh")
    tokenized0 = tokenizer(q_l, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
    tokenized1 = tokenizer(qw_l, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
    tokenized2 = tokenizer(qm_l, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
    tokenized3 = tokenizer(qs_l, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
    tokenized4 = tokenizer(qo_l, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
    labels = torch.tensor(labels, dtype=torch.long)

    tokenized0.to("cuda:0")
    tokenized1.to("cuda:0")
    tokenized2.to("cuda:0")
    tokenized3.to("cuda:0")
    tokenized4.to("cuda:0")

    output0_class, output1_class, output2_class, output3_class, output4_class = model(tokenized0,tokenized1,tokenized2,tokenized3,tokenized4)

    if len(labels)<=10:
        print("less 10")
        continue
    
    # ndcg
    scores_pre_origin = [list(lst).index(max(list(lst))) for lst in output0_class]
    scores_pre_woman = [list(lst).index(max(list(lst))) for lst in output1_class]
    scores_pre_man = [list(lst).index(max(list(lst))) for lst in output2_class]
    score_pre_student = [list(lst).index(max(list(lst))) for lst in output3_class]
    score_pre_old = [list(lst).index(max(list(lst))) for lst in output4_class]

    ndcg_origin_10 = ndcg_score([labels.tolist()],[scores_pre_origin],k=10)
    ndcg_woman_10 = ndcg_score([labels.tolist()],[scores_pre_woman],k=10)
    ndcg_man_10 = ndcg_score([labels.tolist()],[scores_pre_man],k=10)
    ndcg_student_10 = ndcg_score([labels.tolist()],[score_pre_student],k=10)
    ndcg_old_10 = ndcg_score([labels.tolist()],[score_pre_old],k=10)

    ndcg_origin_5 = ndcg_score([labels.tolist()],[scores_pre_origin],k=5)
    ndcg_woman_5 = ndcg_score([labels.tolist()],[scores_pre_woman],k=5)
    ndcg_man_5 = ndcg_score([labels.tolist()],[scores_pre_man],k=5)
    ndcg_student_5 = ndcg_score([labels.tolist()],[score_pre_student],k=5)
    ndcg_old_5 = ndcg_score([labels.tolist()],[score_pre_old],k=5)


    ndcg_origin_10_list.append(ndcg_origin_10)
    ndcg_woman_10_list.append(ndcg_woman_10)
    ndcg_man_10_list.append(ndcg_man_10)
    ndcg_student_10_list.append(ndcg_student_10)
    ndcg_old_10_list.append(ndcg_old_10)


    ndcg_origin_5_list.append(ndcg_origin_5)
    ndcg_woman_5_list.append(ndcg_woman_5)
    ndcg_man_5_list.append(ndcg_man_5)
    ndcg_student_5_list.append(ndcg_student_5)
    ndcg_old_5_list.append(ndcg_old_5)



    # MAP
    enc = OneHotEncoder(categories= [[0,1,2,3,4]])

    y_true_origin=enc.fit_transform(np.array(labels).reshape(-1,1)).toarray()
    y_true_woman=enc.fit_transform(np.array(labels).reshape(-1,1)).toarray()
    y_true_man=enc.fit_transform(np.array(labels).reshape(-1,1)).toarray()
    y_true_student=enc.fit_transform(np.array(labels).reshape(-1,1)).toarray()
    y_true_old=enc.fit_transform(np.array(labels).reshape(-1,1)).toarray()

    map_origin = average_precision_score(y_true_origin.reshape(-1), output0_class.reshape(-1).cpu().detach().numpy())
    map_woman = average_precision_score(y_true_woman.reshape(-1), output1_class.reshape(-1).cpu().detach().numpy())
    map_man = average_precision_score(y_true_man.reshape(-1), output2_class.reshape(-1).cpu().detach().numpy())
    map_student = average_precision_score(y_true_student.reshape(-1), output3_class.reshape(-1).cpu().detach().numpy())
    map_old = average_precision_score(y_true_old.reshape(-1), output4_class.reshape(-1).cpu().detach().numpy())

    map_origin_list.append(map_origin)
    map_woman_list.append(map_woman)
    map_man_list.append(map_man)
    map_student_list.append(map_student)
    map_old_list.append(map_old)

mean_ndcg_5_origin = np.mean(ndcg_origin_5_list)
mean_ndcg_5_woman = np.mean(ndcg_woman_5_list)
mean_ndcg_5_man = np.mean(ndcg_man_5_list)
mean_ndcg_5_student = np.mean(ndcg_student_5_list)
mean_ndcg_5_old = np.mean(ndcg_old_5_list)

arr_ndcg_5 = np.array([[ndcg_origin_5_list]
       ,[ndcg_woman_5_list]
       ,[ndcg_man_5_list]
       ,[ndcg_student_5_list]
       ,[ndcg_old_5_list]])

vndcg_5 = np.mean(np.var(arr_ndcg_5,axis = 0))

mean_ndcg_10_origin = np.mean(ndcg_origin_10_list)
mean_ndcg_10_woman = np.mean(ndcg_woman_10_list)
mean_ndcg_10_man = np.mean(ndcg_man_10_list)
mean_ndcg_10_student = np.mean(ndcg_student_10_list)
mean_ndcg_10_old = np.mean(ndcg_old_10_list)

arr_ndcg_10 = np.array([[ndcg_origin_10_list]
       ,[ndcg_woman_10_list]
       ,[ndcg_man_10_list]
       ,[ndcg_student_10_list]
       ,[ndcg_old_10_list]])
vndcg_10 = np.mean(np.var(arr_ndcg_10,axis = 0))


mean_map_origin = np.mean(map_origin_list)
mean_map_woman = np.mean(map_woman_list)
mean_map_man = np.mean(map_man_list)
mean_map_student = np.mean(map_student_list)
mean_map_old = np.mean(map_old_list)

arr_map = np.array([[map_origin_list]
       ,[map_woman_list]
       ,[map_man_list]
       ,[map_student_list]
       ,[map_old_list]])
vnap = np.mean(np.var(arr_map/ np.mean(arr_map,axis = 0),axis = 0))


print("mean_ndcg_5_origin:{}".format(mean_ndcg_5_origin))
print("mean_ndcg_5_woman:{}".format(mean_ndcg_5_woman))
print("mean_ndcg_5_man:{}".format(mean_ndcg_5_man))
print("mean_ndcg_5_student:{}".format(mean_ndcg_5_student))
print("mean_ndcg_5_old:{}".format(mean_ndcg_5_old))
print("mean_ndcg_10_origin:{}".format(mean_ndcg_10_origin))
print("mean_ndcg_10_woman:{}".format(mean_ndcg_10_woman))
print("mean_ndcg_10_man:{}".format(mean_ndcg_10_man))
print("mean_ndcg_10_student:{}".format(mean_ndcg_10_student))
print("mean_ndcg_10_old:{}".format(mean_ndcg_10_old))
print("mean_map_origin:{}".format(mean_map_origin))
print("mean_map_woman:{}".format(mean_map_woman))
print("mean_map_man:{}".format(mean_map_man))
print("mean_map_student:{}".format(mean_map_student))
print("mean_map_old:{}".format(mean_map_old))
print("vndcg_5:{}".format(vndcg_5))
print("vndcg_10:{}".format(vndcg_10))
print("vnap:{}".format(vnap))
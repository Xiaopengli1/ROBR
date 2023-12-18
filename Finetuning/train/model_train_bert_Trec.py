# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime

"""## Read All Data"""

import pandas as pd
# from sklearn.metrics import ndcg_score
from tqdm import tqdm

data = pd.read_csv("data/Trec04/all_data_en.txt",encoding='utf-8',header=None,sep = '\t',error_bad_lines=False)
data.columns = ["query", "label", "a", "b", "c"]
data = data.fillna("")

data["content"] = data["a"] + data["b"] + data["c"]

del data["a"]
del data["b"]
del data["c"]

query_all = data["query"].apply(lambda x:x).unique()

data_train = data[:200]
data_val = data[200:220]
data_test = data[-220:]

data_val.reset_index(drop =True, inplace=True)
data_test.reset_index(drop =True, inplace=True)

"""## Fine-tune the Model"""

# data_train
# data_val
# data_test

def read_samples(df):
    samples = []
    for i in tqdm(range(len(df))):
        query = df.loc[i]["query"]
        value = df.loc[i]["label"]
        doc   = df.loc[i]["content"]
        value = int(value)
        samples.append(InputExample(texts=[query, doc], label=value))
    return samples

train_samples = read_samples(data_train)
dev_samples = read_samples(data_val)
test_samples = read_samples(data_test)



train_batch_size = 30
num_epochs = 1

model_save_path = 'model_path/Pre_trained_model/en/bert-base-uncased'
# model_save_path = 'model_path/Pre_trained_model/en/ernie-2.0-base-en'
# model_save_path = 'model_path/Pre_trained_model/en/roberta-base-squad2'


#Define our CrossEncoder model_path. We use distilroberta-base as basis and setup it up to predict 3 labels
model = CrossEncoder("model_path/Raw_model/en/bert-base-uncased",num_labels=5, device="cuda", max_length=512)
# model_path = CrossEncoder("model_path/Raw_model/en/ernie-2.0-base-en",num_labels=5, device="cuda", max_length=512)
# model_path = CrossEncoder("model_path/Raw_model/en/roberta-base-squad2",num_labels=5, device="cuda", max_length=512)

#We wrap train_samples, which is a list ot InputExample, in a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

#During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
evaluator = CEBinaryAccuracyEvaluator.from_input_examples(dev_samples)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

# Train the model_path
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=200000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

"""## Fine-tune the model_path"""
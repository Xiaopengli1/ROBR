import pandas as pd
import math
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
from sentence_transformers.readers import InputExample
from sklearn.metrics import ndcg_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from tqdm import tqdm


text = pd.read_csv("data/Trec04/raw_data_full_en.csv")
query = pd.read_csv("data/Trec04/Rewritedquery_en.csv")

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


n = 20

model = CrossEncoder("model_path/Pre_trained_model/en/bert-base-uncased",num_labels=5, device="cuda", max_length=512)


for i in tqdm(range(len(query))):
# for i in tqdm(range(5)):
    original_query = query.iloc[i]["query"]
    woman_query = query.iloc[i]["woman"]
    man_query = query.iloc[i]["old"]
    student_query = query.iloc[i]["student"]
    old_query = query.iloc[i]["old"]

    text_q_origin = text[text["query"]==original_query]

    text_q_mW = text_q_origin.copy(deep=True)
    text_q_mW.loc[:,"query"] = woman_query

    text_q_mM = text_q_origin.copy(deep=True)
    text_q_mM.loc[:,"query"] = man_query

    text_q_student = text_q_origin.copy(deep=True)
    text_q_student.loc[:,"query"] = student_query


    text_q_old = text_q_origin.copy(deep=True)
    text_q_old.loc[:,"query"] = old_query

    scores_origin = model.predict([list(i) for i in zip(text_q_origin["query"].tolist(), text_q_origin["content"].tolist())],apply_softmax=True)
    scores_woman = model.predict([list(i) for i in zip(text_q_mW["query"].tolist(), text_q_mW["content"].tolist())],apply_softmax=True)
    scores_man = model.predict([list(i) for i in zip(text_q_mM["query"].tolist(), text_q_mM["content"].tolist())],apply_softmax=True)
    score_student = model.predict([list(i) for i in zip(text_q_student["query"].tolist(), text_q_student["content"].tolist())],apply_softmax=True)
    score_old = model.predict([list(i) for i in zip(text_q_old["query"].tolist(), text_q_old["content"].tolist())],apply_softmax=True)

    if len(text_q_mM)<=10:
        print("less 10")
        continue
    
    # ndcg
    scores_pre_origin = [list(lst).index(max(list(lst))) for lst in scores_origin]
    scores_pre_woman = [list(lst).index(max(list(lst))) for lst in scores_woman]
    scores_pre_man = [list(lst).index(max(list(lst))) for lst in scores_man]
    score_pre_student = [list(lst).index(max(list(lst))) for lst in score_student]
    score_pre_old = [list(lst).index(max(list(lst))) for lst in score_old]

    ndcg_origin_10 = ndcg_score([text_q_origin.loc[:,"label"].tolist()],[scores_pre_origin],k=10)
    ndcg_woman_10 = ndcg_score([text_q_mW.loc[:,"label"].tolist()],[scores_pre_woman],k=10)
    ndcg_man_10 = ndcg_score([text_q_mM.loc[:,"label"]],[scores_pre_man],k=10)
    ndcg_student_10 = ndcg_score([text_q_student.loc[:,"label"]],[score_pre_student],k=10)
    ndcg_old_10 = ndcg_score([text_q_old.loc[:,"label"].tolist()],[score_pre_old],k=10)

    ndcg_origin_5 = ndcg_score([text_q_origin.loc[:,"label"].tolist()],[scores_pre_origin],k=5)
    ndcg_woman_5 = ndcg_score([text_q_mW.loc[:,"label"].tolist()],[scores_pre_woman],k=5)
    ndcg_man_5 = ndcg_score([text_q_mM.loc[:,"label"]],[scores_pre_man],k=5)
    ndcg_student_5 = ndcg_score([text_q_student.loc[:,"label"]],[score_pre_student],k=5)
    ndcg_old_5 = ndcg_score([text_q_old.loc[:,"label"].tolist()],[score_pre_old],k=5)


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


    y_true_origin=enc.fit_transform(np.array(text_q_origin.loc[:,"label"]).reshape(-1,1)).toarray()
    y_true_woman=enc.fit_transform(np.array(text_q_mW.loc[:,"label"]).reshape(-1,1)).toarray()
    y_true_man=enc.fit_transform(np.array(text_q_mM.loc[:,"label"]).reshape(-1,1)).toarray()
    y_true_student=enc.fit_transform(np.array(text_q_student.loc[:,"label"]).reshape(-1,1)).toarray()
    y_true_old=enc.fit_transform(np.array(text_q_old.loc[:,"label"]).reshape(-1,1)).toarray()

    map_origin = average_precision_score(y_true_origin.reshape(-1), scores_origin.reshape(-1))
    map_woman = average_precision_score(y_true_woman.reshape(-1), scores_woman.reshape(-1))
    map_man = average_precision_score(y_true_man.reshape(-1), scores_man.reshape(-1))
    map_student = average_precision_score(y_true_student.reshape(-1), score_student.reshape(-1))
    map_old = average_precision_score(y_true_old.reshape(-1), score_old.reshape(-1))

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
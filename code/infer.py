#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import gc
from  lightgbm import LGBMClassifier,log_evaluation,early_stopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
import joblib
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

valid = pd.read_pickle('./output_data/valid.pkl')


# In[3]:


w2v_mean_columns = [col for col in valid.columns if "mean_w2v_dif" in col]
w2v_columns = [f'w2v_{i}' for i in range(256)]

def stage2_fe(df,col1,col2,name):
    arr_1 = np.array(df[col1])
    arr_2 = np.array(df[col2])
    cosine_sims = []
    for i in tqdm(range(arr_1.shape[0])):
        vec_1 = arr_1[i]
        vec_2 = arr_2[i]
        cosine_sim = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
        cosine_sims.append(cosine_sim)
    return cosine_sims

w2v_cos_valid = stage2_fe(valid,w2v_columns,w2v_mean_columns,'w2v')   


# In[5]:


tfidf2_mean_columns = [col for col in valid.columns if "mean_cnt_dif" in col]
cnt_columns = [col for col in valid.columns if "tfid_cnt" in col]
all_cnt_columns = cnt_columns[:328]

cnt_cos_valid = stage2_fe(valid,all_cnt_columns,tfidf2_mean_columns,'cnt')   


# In[6]:


valid['w2v_cosine_sims'] = w2v_cos_valid
valid['cnt_cosine_sims'] = cnt_cos_valid


# In[11]:


# 加载模型并进行推理
choose_cols=[col for col in valid.drop(['id','label', 'top_author','author_id'],axis=1).columns]
test_preds = 0
for fold in tqdm(range(5)):
    model_filename = f'./model/lgb_fold_{fold+1}.pkl'
    model = joblib.load(model_filename)
    test_preds += model.predict_proba(valid[choose_cols])[:,1]/5


# In[12]:


with open("./data/ind_test_author_submit.json",encoding='utf-8') as f:
    submission=json.load(f)
    
cnt=0
for id,names in submission.items():
    for name in names:
        submission[id][name]=test_preds[cnt]
        cnt+=1
with open('./output_data/lgb.json', 'w', encoding='utf-8') as f:
    json.dump(submission, f, ensure_ascii=False, indent=4)


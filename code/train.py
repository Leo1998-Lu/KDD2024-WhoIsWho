#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import re
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
warnings.filterwarnings('ignore')
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Load model from HuggingFace Hub
tokenizer_ml = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model_ml = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to('cuda')

def get_sent_embedings(model, tokenizer, sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    encoded_input['input_ids'] = encoded_input['input_ids'].to('cuda')
    if 'token_type_ids' in encoded_input:
        encoded_input['token_type_ids'] = encoded_input['token_type_ids'].to('cuda')
    encoded_input['attention_mask'] = encoded_input['attention_mask'].to('cuda')

    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).detach().cpu().numpy()

    return sentence_embeddings


def get_all_embedings(model, tokenizer, texts, bs=32):
    num_batches = len(texts) // bs + int(len(texts) % bs != 0)
    all_embedings = None
    for i in tqdm(range(num_batches)):
        sentences = texts[i * bs: (i + 1) * bs]
        sentence_embeddings = get_sent_embedings(model, tokenizer, sentences)

        if all_embedings is None:
            all_embedings = sentence_embeddings
        else:
            all_embedings = np.vstack([all_embedings, sentence_embeddings])

    return all_embedings


class Config():
    seed=42
    num_folds=5
    TARGET_NAME ='label'
import random
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
seed_everything(Config.seed)


# In[2]:


import json
with open("./data/train_author.json",encoding='utf-8') as f:
    train_author=json.load(f)
with open("./data/pid_to_info_all.json",encoding='utf-8') as f:
    pid_to_info=json.load(f)
with open("./data/ind_valid_author.json",encoding='utf-8') as f:
    valid_author=json.load(f)
with open("./data/ind_valid_author_submit.json",encoding='utf-8') as f:
    submission=json.load(f)
with open("./data/ind_test_author_filter_public.json",encoding='utf-8') as f:
    test_author=json.load(f)
with open("./data/ind_test_author_submit.json",encoding='utf-8') as f:
    test_submission=json.load(f)


# In[4]:


def clean_text(text):
    # Remove special characters, punctuation, etc.
    # text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.replace('[','')
    text = text.replace(']','')
    text = text.replace('{','')
    text = text.replace('}','')
    text = text.replace("'",'')
    text = text.replace(":",'')
    text = text.lower()

    return text

df=[]
labels=[]
for id,person_info in train_author.items():
    names = person_info['name'].lower()
    for text_id in person_info['normal_data']:
        feat=pid_to_info[text_id]
        author_names = [x['name'] for x in feat['authors'][:5]]
        ans = ','.join(author_names)
        text = str(feat['title'])+' '+ clean_text(str(feat['keywords']))+' '+str(feat['abstract'])#, 'authors', 'venue', 'year']

        if len(str(feat['keywords']))<3:
            keywords = feat['title'].lower()
        else:
            keywords = clean_text(str( feat['keywords']))
            
        try:
            df.append(
                [id,names,feat['authors'],clean_text(str(feat['venue'])),keywords,text,len(feat['title']),len(feat['abstract']),len(feat['keywords']),len(feat['authors'])
                 ,2024-int(feat['year']), len(feat['venue']),ans]
                 )
        except:
            df.append(
                [id,names,feat['authors'],clean_text(str(feat['venue'])),keywords,text,len(feat['title']),len(feat['abstract']),len(feat['keywords']),len(feat['authors'])
                 ,20,10,ans]
                 )
        labels.append(1)
    for text_id in person_info['outliers']:
        feat=pid_to_info[text_id]
        author_names = [x['name'] for x in feat['authors'][:5]]
        ans = ','.join(author_names)
        text = feat['title']+ clean_text(str(feat['keywords'])) +feat['abstract']#, 'authors', 'venue', 'year']
   
        if len(str(feat['keywords']))<3:
            keywords = feat['title'].lower()
        else:
            keywords = clean_text(str( feat['keywords']))
            
        try:
            df.append(
                [id,names,feat['authors'],clean_text(str(feat['venue'])),keywords,text,len(feat['title']),len(feat['abstract']),len(feat['keywords']),len(feat['authors'])
                 ,2024-int(feat['year']), len(feat['venue']),ans]
                 )
        except:
            df.append(
                [id,names,feat['authors'],clean_text(str(feat['venue'])),keywords,text,len(feat['title']),len(feat['abstract']),len(feat['keywords']),len(feat['authors'])
                 ,20,10,ans]
                 )
        labels.append(1)
        try:
            df.append(
                [id,names, feat['authors'],clean_text(str(feat['venue'])),keywords,text,len(feat['title']),len(feat['abstract']),len(feat['keywords']),len(feat['authors'])
                 ,2024-int(feat['year']), len(feat['venue']),ans]
                 )
        except:
            df.append(
                [id,names, feat['authors'],clean_text(str(feat['venue'])),keywords,text,len(feat['title']),len(feat['abstract']),len(feat['keywords']),len(feat['authors'])
                 ,20,10,ans]
                 )
        labels.append(0)   
labels=np.array(labels)
df=pd.DataFrame(df)
df['label']=labels
print(f"df.shape:{df.shape},labels.shape:{labels.shape}")
print(f"np.mean(labels):{np.mean(labels)}")
df.columns = ['author_id','name','author','venue','keywords','text','x1','x2','x3','x4','x5','x6','top_author','label']
df.head()


# In[6]:


test_feats=[]
for id,person_info in test_author.items():
    names = person_info['name'].lower()
    for text_id in person_info['papers']:
        feat=pid_to_info[text_id]
        author_names = [x['name'] for x in feat['authors'][:5]]
        ans = ', '.join(author_names)
        text = str(feat['title'])+' '+ clean_text(str(feat['keywords']))+' '+str(feat['abstract'])

        if len(str(feat['keywords']))<3:
            keywords = feat['title'].lower()
        else:
            keywords = clean_text(str(feat['keywords']))
        try:
            test_feats.append(
                [id,names, feat['authors'],clean_text(str(feat['venue'])),keywords,text, len(feat['title']),len(feat['abstract']),len(feat['keywords']),len(feat['authors'])
                 ,2024-int(feat['year']), len(feat['venue']),ans]
                 )
        except:
            test_feats.append(
                [id,names, feat['authors'],clean_text(str(feat['venue'])),keywords,text, len(feat['title']),len(feat['abstract']),len(feat['keywords']),len(feat['authors'])
                 ,20,10,ans]
                 )

test_feats=pd.DataFrame(test_feats)
print(f"valid_feats.shape:{test_feats.shape}")
test_feats.columns = ['author_id','name','author','venue','keywords','text','x1','x2','x3','x4','x5','x6','top_author']
test_feats.head()


# In[8]:


from sklearn.preprocessing import LabelEncoder
def gg(df):
    author_ex = [] 
    for x,y in zip(df['name'],df['top_author']):
        if x in y:
            author_ex.append(1)
        else:
            author_ex.append(0)
    df['author_ex'] = author_ex
        
    for x in ['author','keywords','text','top_author']:
        df[f'{x}_len'] = df[x].apply(lambda x: len(str(x)))
    df['text'] = df['text'].apply(lambda x: x.lower())
    df['top_author'] = df['top_author'].apply(lambda x: x.lower())
    df['name'] = df['name'].apply(lambda x: x.lower())
    le = LabelEncoder()
    df['name'] = le.fit_transform(df['author_id'])
    df['venue'] = le.fit_transform(df['venue'])
    df['venue'] = df['venue'].apply(lambda x: str(x).lower())

    return df
train = gg(df)
valid = gg(test_feats)
valid


# In[9]:


df = pd.concat([train,valid],axis=0).reset_index(drop=True)
df['id'] = df.index
df


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# TfidfVectorizer parameter
vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            strip_accents='unicode',
            analyzer = 'word',
            ngram_range=(1,4),
            min_df=0.05,
            max_df=0.95,
            sublinear_tf=True,max_features=100
)

X = vectorizer.fit_transform(df['text'])
X = X.toarray()
tfidf_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
print(10*'*','Stage1 Done',10*'*')
vectorizer2 = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            strip_accents='unicode',
            analyzer = 'word',
            ngram_range=(1,3),
            min_df=0.05,
            max_df=0.95,
            sublinear_tf=True,max_features=100
)
X2 = vectorizer2.fit_transform(df['keywords'])
X2 = X2.toarray()
tfidf_df2 = pd.DataFrame(X2, columns=[f'keywords_{i}' for i in range(X2.shape[1])])
print(10*'*','Stage2 Done',10*'*')

vectorizer3 = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            strip_accents='unicode',
            analyzer = 'word',
            ngram_range=(1,2),
            min_df=0.05,
            max_df=0.95,
            sublinear_tf=True
)
X3 = vectorizer3.fit_transform(df['venue'])
X3 = X3.toarray()
tfidf_df3 = pd.DataFrame(X3, columns=[f'venue_{i}' for i in range(X3.shape[1])])
print(10*'*','Stage3 Done',10*'*')


# In[11]:


vectorizer_cnt = CountVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            strip_accents='unicode',
            analyzer = 'word',
            ngram_range=(1,3),
            min_df=0.10,
            max_df=0.85,max_features=100
)
print(10*'*','Stage1 CountVectorizer Starting',10*'*')
train_tfid = vectorizer_cnt.fit_transform([i for i in df['text']])
train_tfid  = train_tfid.toarray()
cnt_df = pd.DataFrame(train_tfid, columns=[f'tfid_cnt_{i}' for i in range(train_tfid.shape[1])])

vectorizer_cnt2 = CountVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            strip_accents='unicode',
            analyzer = 'word',
            ngram_range=(2,3),
            min_df=0.10,
            max_df=0.85,max_features=100
)
print(10*'*','Stage2 CountVectorizer Starting',10*'*')
train_tfid2 = vectorizer_cnt2.fit_transform([i for i in df['keywords']])
train_tfid2 = train_tfid2.toarray()
cnt_df2 = pd.DataFrame(train_tfid2, columns=[f'tfid_cnt2_{i}' for i in range(train_tfid2.shape[1])])

vectorizer_cnt3 = CountVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            strip_accents='unicode',
            analyzer = 'word',
            ngram_range=(1,2),
            min_df=0.10,
            max_df=0.85,
)
print(10*'*','Stage3 CountVectorizer Starting',10*'*')
train_tfid3 = vectorizer_cnt3.fit_transform([i for i in df['venue']])
train_tfid3 = train_tfid3.toarray()
cnt_df3 = pd.DataFrame(train_tfid3, columns=[f'tfid_cnt3_{i}' for i in range(train_tfid3.shape[1])])



# In[12]:


vectorizer_cnt4 = CountVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            strip_accents='unicode',
            analyzer = 'word',
            ngram_range=(1,2),
            min_df=0.10,
            max_df=0.85,
)
train_tfid4 = vectorizer_cnt4.fit_transform([i for i in df['top_author']])
train_tfid4 = train_tfid4.toarray()
cnt_df4 = pd.DataFrame(train_tfid4, columns=[f'tfid_cnt4_{i}' for i in range(train_tfid4.shape[1])])


# In[13]:


from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# 对文本进行预处理,分词
processed_texts = [simple_preprocess(text) for text in df['text']]
size = 256
# 训练 Word2Vec 模型
model = Word2Vec(processed_texts, vector_size=size, window=5, min_count=2, workers=16)# 提取每个文本的 Word2Vec 特征向量
X = []
for text in processed_texts:
    vector = np.zeros(size)
    count = 0
    for word in text:
        if word in model.wv:
            vector += model.wv[word]
            count += 1
    if count > 0:
        vector /= count
    X.append(vector)
w2v = pd.DataFrame(X, columns=[f'w2v_{i}' for i in range(size)])


# In[14]:


df = pd.concat([df.reset_index(drop=True),tfidf_df,tfidf_df2,tfidf_df3,cnt_df,cnt_df2,cnt_df3,cnt_df4,w2v],axis=1)


# In[15]:


import pandas as pd
from collections import Counter
import string
import gc
train = df[~df['label'].isna()].reset_index(drop=True)
valid = df[df['label'].isna()].reset_index(drop=True)
def preprocess_text(text):
    # 去除标点符号
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 转换为小写
    text = text.lower()
    text = text.replace('the','')
    text = text.replace('of','')
    text = text.replace('a','')
    text = text.replace('an','')
    text = text.replace('how','')
    text = text.replace('what','')
    text = text.replace('which','')
    text = text.replace('why','')
    text = text.replace('where','')
    text = text.replace('who','')
    text = text.replace('we','')
    text = text.replace('are','')
    text = text.replace('is','')
    text = text.replace('by','')
    text = text.replace('base','')
    text = text.replace('on','')
    text = text.replace('in','')
    return text

def calculate_overlap_score(text, top100_words):
    # 将文本拆分为单词
    words = text.split()
    # 计算文本中出现在top100单词中的单词数量
    overlap_count = sum(1 for word in words if word in top100_words)
    # 计算重合度指标
    overlap_score = overlap_count / len(words) if len(words) > 0 else 0
    return overlap_score


# 按作者分组
def get_overlap_score(df,col):
    df[col] = df[col].astype(str).apply(preprocess_text)
    grouped = df.groupby("name")
    
    # 计算每个作者在label为1的样本中出现频率最高的前100个关键词
    top100_words = {}
    for author, group in tqdm(grouped):
        # if mode =='train':
        #     author_label_1 = group[group["label"] == 1]
        author_label_1_text = " ".join(group[col])
        words = author_label_1_text.split()
        word_counts = Counter(words)
        top100 = [word for word, _ in word_counts.most_common(100)]
        top100_words[author] = top100

    # 计算当前样本文本与对应作者的top100关键词的重合度指标
    cnt = 0
    for x,y in zip(df['name'],df[col]):
        cnt  += 1
        overlap_score = calculate_overlap_score(y, top100_words[x])
        df.loc[cnt-1,f'{col}_overlap_score'] = overlap_score
    return df
for col in ['text','keywords']:
    train = get_overlap_score(train,col)
    valid  = get_overlap_score(valid,col)


# In[16]:


def stage1_fe(df):

    df["text_content_cnt"] = df['text'].apply(lambda x: len(x.split('.')))
    df["text_sentence_cnt"] = df['text'].apply(lambda x: len(x.split(',')))
    df["text_word_cnt"] = df['text'].apply(lambda x: len(x.split(' ')))


    
    for col in ['x5','venue','id']:
        df[f'name_{col}_nunique'] = df.groupby(['name'])[col].transform('nunique')
        df[f'name_{col}_count'] = df.groupby(['name'])[col].transform('count')
    for col in ['name','venue','id']:
        df[f'year_{col}_nunique'] = df.groupby(['x5'])[col].transform('nunique')
        df[f'year_{col}_count'] = df.groupby(['x5'])[col].transform('count')
    
    name_dict = df['name'].value_counts().to_dict()
    df['name_count'] = df['name'].map(name_dict)
    venue_dict = df['venue'].value_counts().to_dict()
    df['venue_count'] = df['venue'].map(venue_dict)
    
    for col in tqdm(['x1', 'x2', 'x3', 'x4', 'x5','x6',"text_content_cnt","text_sentence_cnt","text_word_cnt"]):#, 'author_len', 'keywords_len', 'text_len','org'
        for meth in ['mean','max','min','std','median','skew']:
            df[f'{col}_{meth}_name'] = df.groupby(['name'])[col].transform(meth)
            
    for col in tqdm(['x1', 'x2', 'x3', 'x4', 'x6',"text_content_cnt","text_sentence_cnt","text_word_cnt"]):#, 'author_len', 'keywords_len', 'text_len','org'
        for meth in ['mean','max','min','std','median','skew']:
            df[f'year_{col}_{meth}'] = df.groupby(['x5'])[col].transform(meth)

    for col in ["text_content_cnt","text_sentence_cnt","text_word_cnt"]:
        for meth in ['mean','max','min']:
            df[f'{col}_{meth}_name_dif'] = df[f'{col}_{meth}_name'] - df[col]
            df[f'{col}_{meth}_name_ratio'] = df[col]/(1+df[f'{col}_{meth}_name'])
            
            df[f'year_{col}_{meth}_dif'] = df[f'year_{col}_{meth}'] - df[col]
            df[f'year_{col}_{meth}_ratio'] = df[col]/(1+df[f'year_{col}_{meth}'])
        
            
    
    for col in tqdm(list(tfidf_df.columns)+list(tfidf_df2.columns)+list(tfidf_df3.columns)):
         for meth in ['mean','max','min']:
            df[f'{col}_{meth}_tfidf_dif'] = df.groupby(['name'])[col].transform(meth)-df[col]

    
    for col in tqdm(list(cnt_df.columns)+list(cnt_df2.columns)+list(cnt_df3.columns)+list(cnt_df4.columns)):
         for meth in ['mean','max','min']:
            df[f'{col}_{meth}_cnt_dif'] = df.groupby(['name'])[col].transform(meth)-df[col]
    for col in tqdm(list(w2v.columns)):
         for meth in ['mean','max','min']:
            df[f'{col}_{meth}_w2v_dif'] = df.groupby(['name'])[col].transform(meth)-df[col]
             
    # for col in tqdm([f'A_emb_{i}' for i in range(100)]):
    #      for meth in ['mean','max','min']:
    #         df[f'A_emb_{col}_{meth}'] =df.groupby(['name'])[col].transform(meth)-df[col]
             
    # for col in tqdm(list(emb_deberta.columns)):
    #      for meth in ['mean','max','min']:
    #         df[f'{col}_{meth}_deberta_dif'] = df.groupby(['name'])[col].transform(meth)-df[col]
    
   
    return df
train = stage1_fe(train)
valid = stage1_fe(valid)
del tfidf_df,tfidf_df2,tfidf_df3,cnt_df,cnt_df2,cnt_df3,w2v
gc.collect()


# In[17]:


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


# In[18]:


train = train.drop(['author','text','keywords','venue'],axis=1)
valid = valid.drop(['author','text','keywords','venue'],axis=1)


# In[19]:


from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df
train.to_pickle('./output_data/train.pkl')
valid.to_pickle('./output_data/valid.pkl')


# # Train

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
warnings.filterwarnings('ignore')
train = pd.read_pickle('./output_data/train.pkl')
valid = pd.read_pickle('./output_data/valid.pkl')


# In[34]:


w2v_mean_columns = [col for col in train.columns if "mean_w2v_dif" in col]
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

w2v_cos_train = stage2_fe(train,w2v_columns,w2v_mean_columns,'w2v')
w2v_cos_valid = stage2_fe(valid,w2v_columns,w2v_mean_columns,'w2v')   


# In[35]:


tfidf2_mean_columns = [col for col in train.columns if "mean_cnt_dif" in col]
cnt_columns = [col for col in train.columns if "tfid_cnt" in col]
all_cnt_columns = cnt_columns[:328]

cnt_cos_train = stage2_fe(train,all_cnt_columns,tfidf2_mean_columns,'cnt')
cnt_cos_valid = stage2_fe(valid,all_cnt_columns,tfidf2_mean_columns,'cnt')   


# In[36]:


train['w2v_cosine_sims'] = w2v_cos_train
valid['w2v_cosine_sims'] = w2v_cos_valid
train['cnt_cosine_sims'] = cnt_cos_train
valid['cnt_cosine_sims'] = cnt_cos_valid


# In[57]:


import joblib
choose_cols=[col for col in valid.drop(['id','label', 'top_author','author_id'],axis=1).columns]
print(len(choose_cols))
N_FOLDS = 5
good_feats = []
def fit_and_predict(model,train=train,test=valid):
    X=train[choose_cols].copy()
    y=train['label'].copy()
    test_X=test[choose_cols].copy()
    oof_pred_pro=np.zeros((len(X),2))
    test_pred_pro=np.zeros((N_FOLDS,len(test_X),2))

    skf = StratifiedKFold(n_splits=N_FOLDS,random_state=42, shuffle=True)

    for fold, (train_index, valid_index) in (enumerate(skf.split(X, X['name']))):
        print(f"fold:{fold}")

        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]


        model.fit(X_train,y_train,eval_set=[(X_valid, y_valid)],
                  callbacks=[log_evaluation(100),early_stopping(100)]
                 )
        
        oof_pred_pro[valid_index]=model.predict_proba(X_valid)
        #将数据分批次进行预测.
        test_pred_pro[fold]=model.predict_proba(test_X)

        # 保存模型
        model_filename = f'lgb_fold_{fold+1}.pkl'
        joblib.dump(model, model_filename)
        print(f"Model saved to {model_filename}")

        # 查看特征重要性排名 (从高到低)
        feat_imp = pd.Series(model.booster_.feature_importance('gain'), index=X_train.columns).sort_values(ascending=False)[:30]
        print(feat_imp)
        
    print(f"roc_auc:{roc_auc_score(y.values,oof_pred_pro[:,1])}")
    # 获取特征重要性
    
    return oof_pred_pro,test_pred_pro,good_feats

lgb_params={
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators":10000,
    "colsample_bytree": 0.2,
    "colsample_bynode": 0.2,
    "verbose": -1,
    "random_state": 42,
    "reg_alpha": 0.1,
    "reg_lambda": 10,
    "extra_trees":True,
    'num_leaves':127,
    "verbose": -1,
    "max_bin":225,
    "class_weight":'balanced',
    }


lgb_oof_pred_pro,lgb_test_pred_pro,good_feats=fit_and_predict(model= LGBMClassifier(**lgb_params),train=train,test=valid)
test_preds=lgb_test_pred_pro.mean(axis=0)[:,1]


# In[26]:


with open("./data/ind_test_author_submit.json",encoding='utf-8') as f:
    submission=json.load(f)
    
cnt=0
for id,names in submission.items():
    for name in names:
        submission[id][name]=test_preds[cnt]
        cnt+=1
with open('./output_data/lgb.json', 'w', encoding='utf-8') as f:
    json.dump(submission, f, ensure_ascii=False, indent=4)


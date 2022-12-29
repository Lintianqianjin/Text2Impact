import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse import save_npz

import json
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

import torch

from transformers import AutoTokenizer, AutoModel

data_dir = '2022-06-02'
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
lemmatize = lambda sent: ' '.join(tokenizer.convert_ids_to_tokens(
                            tokenizer([sent], padding=False, truncation=False, 
                                      return_tensors=None,return_attention_mask=False,
                                      add_special_tokens=False)['input_ids'][0]
                        ))


#data = pd.read_csv(f'scibert_encoded_papers/needed_paper_metadata_{data_dir}.csv')

data = pd.read_csv(f'needed_paper_metadata_{data_dir}.csv')

data.fillna(' ',inplace=True)
data['concat_title_abs'] =data[['title', 'abstract']].agg('. '.join, axis=1)

data.drop(columns=['title','abstract'],inplace=True)

data['concat_title_abs'] = data['concat_title_abs'].apply(lemmatize)


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = data['concat_title_abs'].to_list()
vectorizer = TfidfVectorizer(min_df=3)
X = vectorizer.fit_transform(corpus)

print(f"number of words: {len(vectorizer.get_feature_names())}")

now = []
for i in tqdm(range(100)):
    vectorizer = TfidfVectorizer(min_df=i)
    X = vectorizer.fit_transform(corpus)
    now.append(len(vectorizer.get_feature_names()))

vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)

df = pd.DataFrame.sparse.from_spmatrix(X,columns=vectorizer.get_feature_names(),index=data['cord_uid'].to_list())


save_npz(f'values_{data_dir}_scibert_token.npz', X)

with open(f'index_{data_dir}_scibert_token','w',encoding='utf-8') as fw:
    fw.write(json.dumps(df.index.to_list(),indent=1))

with open(f'column_{data_dir}_scibert_token','w',encoding='utf-8') as fw:
    fw.write(json.dumps(df.columns.to_list(),indent=1))
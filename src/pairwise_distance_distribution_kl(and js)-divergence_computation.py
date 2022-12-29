import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity, paired_distances
import numpy as np
np.random.seed(0)

from tqdm import tqdm
import random

# libraries & dataset
import seaborn as sns
import matplotlib.pyplot as plt
# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
sns.set_theme(style="white", palette="pastel")
import scipy
import json

from scipy import sparse

import time
import gc

data_dir = '2022-06-02'

title_embs_df = pd.read_csv(f'title_embs_df_{data_dir}.csv',index_col=0)
abstract_embs_df = pd.read_csv(f'abstract_embs_df_{data_dir}.csv',index_col=0)
paper_emb = title_embs_df.join(abstract_embs_df,lsuffix='_ti',rsuffix='_ab')

citation_net = pd.read_csv(f'citation_net_{data_dir}.tsv',sep='\t')

citing_pair = citation_net.loc[~(citation_net['citing_uid'].isna() | citation_net['cited_uid'].isna()) ][['citing_uid','cited_uid']]

citing_pair = citing_pair.drop_duplicates()

citing_pair_set = list(map(lambda x:tuple(x),citing_pair.values))

neg = np.random.choice(list(paper_emb.index), size=citing_pair.shape[0])

non_citing_pairs = [(citing_uid,neg[i]) for i, citing_uid in enumerate(citing_pair['citing_uid'])]

for x, y in tqdm(zip(citing_pair_set, non_citing_pairs)):
    if x[0] != y[0]: break
else:
    print('Order Correct')


del title_embs_df
del abstract_embs_df
del paper_emb
gc.collect()


embedding_dict = dict()



for method in ['tfidf', 'glove','scibert']:

    # SCIBERT Embedding
    if method == 'scibert':
        print('scibert')
        title_embs_df = pd.read_csv(f'title_embs_df_{data_dir}.csv',index_col=0)
        abstract_embs_df = pd.read_csv(f'abstract_embs_df_{data_dir}.csv',index_col=0)
        paper_emb = title_embs_df.join(abstract_embs_df,lsuffix='_ti',rsuffix='_ab')
        uid_map = {uid:i for i,uid in enumerate(title_embs_df.index)}
        citing_uid_embeds = paper_emb.values[list(map(lambda x:uid_map[x],citing_pair['citing_uid'].tolist()))]
        cited_uid_embeds = paper_emb.values[list(map(lambda x:uid_map[x],citing_pair['cited_uid'].tolist()))]
        fake_citing_uid_embeds = paper_emb.values[list(map(lambda x:uid_map[x[0]],non_citing_pairs))]
        fake_cited_uid_embeds = paper_emb.values[list(map(lambda x:uid_map[x[1]],non_citing_pairs))]
        embedding_dict['scibert'] = [citing_uid_embeds, cited_uid_embeds, fake_citing_uid_embeds, fake_cited_uid_embeds]

    # Glove Embedding
    if method == 'glove':
        print('glove')
        title_embs_df = pd.read_csv(f'title_embs_df_glove_{data_dir}.csv',index_col=0)
        abstract_embs_df = pd.read_csv(f'abstract_embs_df_glove_{data_dir}.csv',index_col=0)
        paper_emb = title_embs_df.join(abstract_embs_df,lsuffix='_ti',rsuffix='_ab')
        uid_map = {uid:i for i,uid in enumerate(title_embs_df.index)}
        citing_uid_embeds = paper_emb.values[list(map(lambda x:uid_map[x],citing_pair['citing_uid'].tolist()))]
        cited_uid_embeds = paper_emb.values[list(map(lambda x:uid_map[x],citing_pair['cited_uid'].tolist()))]
        fake_citing_uid_embeds = paper_emb.values[list(map(lambda x:uid_map[x[0]],non_citing_pairs))]
        fake_cited_uid_embeds = paper_emb.values[list(map(lambda x:uid_map[x[1]],non_citing_pairs))]
        embedding_dict['glove'] = [citing_uid_embeds, cited_uid_embeds, fake_citing_uid_embeds, fake_cited_uid_embeds]

    # TF-IDF Embedding
    if method == 'tfidf':
        print('tfidf')
        values = scipy.sparse.load_npz(f'values_{data_dir}_scibert_token.npz')
        index = json.load(open(f"index_{data_dir}_scibert_token",'r',encoding='utf-8'))
        column = json.load(open(f"column_{data_dir}_scibert_token",'r',encoding='utf-8'))
        # paper_emb = pd.DataFrame.sparse.from_spmatrix(values,columns=column,index=index)
        uid_map = {uid:i for i,uid in enumerate(index)}
        citing_uid_embeds = values[list(map(lambda x:uid_map[x],citing_pair['citing_uid'].tolist()))]
        cited_uid_embeds = values[list(map(lambda x:uid_map[x],citing_pair['cited_uid'].tolist()))]
        fake_citing_uid_embeds = values[list(map(lambda x:uid_map[x[0]],non_citing_pairs))]
        fake_cited_uid_embeds = values[list(map(lambda x:uid_map[x[1]],non_citing_pairs))]
        embedding_dict['tfidf'] = [citing_uid_embeds, cited_uid_embeds, fake_citing_uid_embeds, fake_cited_uid_embeds]

citing_pair_dis_dict = dict()

for method, embeddings in embedding_dict.items():
    citing_uid_embeds, cited_uid_embeds, fake_citing_uid_embeds, fake_cited_uid_embeds = embeddings
    citing_pair_dis = {'l2':[],'cos':[]}
    for i in tqdm(range(citing_uid_embeds.shape[0])):
        citing_uid_emb, cited_uid_emb = citing_uid_embeds[i], cited_uid_embeds[i]
        l2_dis = np.linalg.norm((citing_uid_emb-cited_uid_emb) if isinstance(cited_uid_emb, np.ndarray) else (citing_uid_emb-cited_uid_emb).todense())
        if isinstance(cited_uid_emb, np.ndarray):
            cos_sim = (citing_uid_emb.dot(cited_uid_emb.T)) / (np.sqrt(citing_uid_emb.dot(citing_uid_emb.T)) * np.sqrt(cited_uid_emb.dot(cited_uid_emb.T)) +1e-10)
        else:
            cos_sim = np.array((citing_uid_emb.dot(cited_uid_emb.T)).todense())[0][0] / (np.array(np.sqrt(citing_uid_emb.dot(citing_uid_emb.T).todense()))[0][0] * np.array(np.sqrt(cited_uid_emb.dot(cited_uid_emb.T).todense()))[0][0] +1e-10)
        citing_pair_dis['l2'].append(l2_dis)
        citing_pair_dis['cos'].append(cos_sim)
    
    non_citing_pair_dis = {'l2':[],'cos':[]}
    for i in tqdm(range(fake_citing_uid_embeds.shape[0])):
        citing_uid_emb, cited_uid_emb = fake_citing_uid_embeds[i], fake_cited_uid_embeds[i]
        l2_dis = np.linalg.norm((citing_uid_emb-cited_uid_emb) if isinstance(cited_uid_emb, np.ndarray) else (citing_uid_emb-cited_uid_emb).todense())
        if isinstance(cited_uid_emb, np.ndarray):
            cos_sim = (citing_uid_emb.dot(cited_uid_emb.T)) / (np.sqrt(citing_uid_emb.dot(citing_uid_emb.T)) * np.sqrt(cited_uid_emb.dot(cited_uid_emb.T)) +1e-10)
        else:
            cos_sim = np.array((citing_uid_emb.dot(cited_uid_emb.T)).todense())[0][0] / (np.array(np.sqrt(citing_uid_emb.dot(citing_uid_emb.T).todense()))[0][0] * np.array(np.sqrt(cited_uid_emb.dot(cited_uid_emb.T).todense()))[0][0] +1e-10)
        non_citing_pair_dis['l2'].append(l2_dis)
        non_citing_pair_dis['cos'].append(cos_sim)
    
    citing_pair_dis_dict[method] = [citing_pair_dis, non_citing_pair_dis]




for method in ['tfidf', 'glove','scibert']:
    # citing 的 cos 转换为在 [-1,1]的离散分布
    cnts_0, bins = np.histogram(np.array(citing_pair_dis_dict[method][0]['cos']),bins=200,range=(-1,1))
    cnts_0 = cnts_0/len(np.array(citing_pair_dis_dict[method][0]['cos']))
    # non - citing 的 cos 转换为在 [-1,1]的离散分布
    print(method+"max",np.array(citing_pair_dis_dict[method][0]['cos']).max())
    print(method+"min",np.array(citing_pair_dis_dict[method][0]['cos']).min())

    print(method+"range",np.array(citing_pair_dis_dict[method][0]['cos']).max()-np.array(citing_pair_dis_dict[method][0]['cos']).min())


    cnts_1, bins = np.histogram(np.array(citing_pair_dis_dict[method][1]['cos']),bins=200,range=(-1,1))
    cnts_1 = (cnts_1+1)/len(np.array(citing_pair_dis_dict[method][1]['cos']))

    # KL entropy
    print(method+"citing",scipy.stats.entropy(cnts_0))
    print(method+"non-citing",scipy.stats.entropy(cnts_1))
    # # JS 
    # M = (cnts_0+cnts_1)/2

    # print(method,0.5*scipy.stats.entropy(cnts_0,M)+0.5*scipy.stats.entropy(cnts_1,M))

import json
out_file = open("pairwise.json", "w") 
  
json.dump(citing_pair_dis_dict, out_file) 
  
out_file.close()
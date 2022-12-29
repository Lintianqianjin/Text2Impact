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

citing_pair_set = set(map(lambda x:tuple(x),citing_pair.values))

non_citing_pairs = list(filter(lambda x: x not in citing_pair_set and x[0]!=x[1],map(lambda x:tuple(x),np.random.choice(list(paper_emb.index),size=147248*2).reshape(-1,2))))


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


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14,5))

cols = list(citing_pair_dis_dict.keys())
rows = ['Euclidean Distance', 'Cosine Similarity']
pad = 5 # in points
for ax, col in zip(axes[0], cols):
    ax.annotate(col.upper(), xy=(0.5, 1), xytext=(0, 2*pad), fontsize=16,
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline')

for idx, (method, dist) in enumerate(citing_pair_dis_dict.items()):
    
    citing_pair_dis, non_citing_pair_dis = dist

    zeros_cnt_citation = (np.array(citing_pair_dis['cos'])<=0).sum()
    zeros_cnt_fake = (np.array(non_citing_pair_dis['cos'])<=0).sum()

    sns.kdeplot(non_citing_pair_dis['l2'], fill=True, color="salmon", label='Pairs without citation', ax=axes[0][idx])
    sns.kdeplot(citing_pair_dis['l2'], fill=True, color="skyblue", label='Pairs with citation', ax=axes[0][idx])
#     loc = 'upper left' if method in ['tfidf'] else 'upper right'
#     axes[0][idx].legend(fontsize=12, loc=loc)
    axes[0][idx].set_xlabel('Euclidean Distance',fontsize=16)
    axes[0][idx].set_ylabel('Density',fontsize=16)

    sns.kdeplot(non_citing_pair_dis['cos'], fill=True, color="salmon",label='Pairs without citation', ax=axes[1][idx])
    sns.kdeplot(citing_pair_dis['cos'], fill=True, color="skyblue",label='Pairs with citation', ax=axes[1][idx])
    axes[1][idx].set_xlabel('Cosine Similarity',fontsize=16)
    axes[1][idx].set_ylabel('Density',fontsize=16)
#     loc = 'upper right' if method in ['tfidf'] else 'upper left'
#     axes[1][idx].legend(fontsize=12, loc=loc)


handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=16)
    
fig.tight_layout()
fig.subplots_adjust(bottom=0.25)
fig.savefig('citing_pair.jpg',dpi=500)



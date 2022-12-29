import pandas as pd
import numpy as np
# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import os
from tqdm import tqdm

data_dir = '2022-06-02'
needed_metadata = pd.read_csv(f'needed_paper_metadata_{data_dir}.csv')
needed_metadata = needed_metadata.set_index('cord_uid')

# scibert 
import torch
from transformers import AutoTokenizer, AutoModel
def scibert():
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    return tokenizer,model

class texts(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, texts_list):
        """
        """
        self.texts_list = texts_list

    def __len__(self):
        return len(self.texts_list)

    def __getitem__(self, idx):
        
        return self.texts_list[idx]

def texts_dataloader(dataset,batch_size,tokenizer):
    def collate_fn(batch_texts):
        texts = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt",max_length=512)
        input_ids = texts['input_ids']
        attention_mask = texts['attention_mask']
        token_type_ids = texts['token_type_ids']
        return input_ids,attention_mask,token_type_ids
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=collate_fn,drop_last=False)


tokenizer,model = scibert()
# title_dataset = texts(needed_metadata['title'].tolist())
# title_dataloader = texts_dataloader(title_dataset,batch_size=64,tokenizer=tokenizer)
# title_embs = []
# with torch.no_grad():
#     model.cuda()
#     model.eval()
#     for step,(input_ids,attention_mask,token_type_ids) in tqdm(enumerate(title_dataloader)):
#         output = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), token_type_ids=token_type_ids.cuda())
# #         print(output.shape)
#         titles_emb = output.last_hidden_state.mean(dim=1).detach()
# #         print(titles_emb.shape)
#         title_embs.append(titles_emb.cpu().numpy())
# #         print(titles_emb.flatten().cpu().numpy().shape)
# #         break
# #         break


# title_embs_array = np.vstack(title_embs)
# title_embs_df = pd.DataFrame(data=title_embs_array,columns=['emb'+str(i) for i in range(768)],index=needed_metadata.index)

# title_embs_df.to_csv(f'title_embs_df_{data_dir}.csv')


needed_metadata = needed_metadata.fillna('')
abstract_dataset = texts(needed_metadata['abstract'].tolist())
abstract_dataloader = texts_dataloader(abstract_dataset,batch_size=4,tokenizer=tokenizer)

abstract_embs = []
with torch.no_grad():
    model.cuda()
    model.eval()
    for step,(input_ids,attention_mask,token_type_ids) in tqdm(enumerate(abstract_dataloader)):
        output = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), token_type_ids=token_type_ids.cuda())
        abstract_emb = output.last_hidden_state.mean(dim=1).detach()
        abstract_embs.append(abstract_emb.cpu().numpy())

abstract_embs_array = np.vstack(abstract_embs)
abstract_embs_df = pd.DataFrame(data=abstract_embs_array,columns=['emb'+str(i) for i in range(768)],index=needed_metadata.index)


abstract_embs_df.to_csv(f'abstract_embs_df_{data_dir}.csv')
null

#Run
"""
sbatch --job-name=CFX --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/recommenders_training.py"
"""
#   
# # Imports

import pandas as pd
import numpy as np
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
export_dir = os.getcwd()
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

from pathlib import Path
import pickle
import torch
from torch import nn, optim, Tensor
import torch.nn as nn
import torch.nn.functional as F
import optuna
import logging
import matplotlib.pyplot as plt

from recommenders_architecture import *
from help_functions import *
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul

from torch_geometric.data import download_url, extract_zip
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from scipy import sparse

import os
import argparse

data_name = "ML1M" ### Can be ML1M, Yahoo, Pinterest
recommender_name = "MLP" ## Can be MLP, VAE, MLP_model, GMF_model, NCF, LightGCN

DP_DIR = Path("processed_data", data_name) 
export_dir = Path(os.getcwd())
files_path = Path(export_dir, DP_DIR)
checkpoints_path = Path(export_dir, "checkpoints")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
# print(f'device is set to {device}')

output_type_dict = {
    "VAE":"multiple",
    "MLP":"single",
    "LightGCN":"single" #changed
}

num_users_dict = {
    "ML1M":6037,
    "Yahoo":13797, 
    "Pinterest":19155
}

num_items_dict = {
    "ML1M":3381,
    "Yahoo":4604, 
    "Pinterest":9362
}

train_losses_dict = {}
test_losses_dict = {}
HR10_dict = {}

ITERATIONS = 5000
EPOCHS = 20

BATCH_SIZE = 1024
LR = 1e-3
ITERS_PER_EVAL = 200
ITERS_PER_LR_DECAY = 200
LAMBDA = 1e-6


def MLP_objective(trial):
    
    lr = trial.suggest_float('learning_rate', 0.001, 0.01)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    beta = trial.suggest_float('beta', 0, 4) # hyperparameter that weights the different loss terms
    epochs = 20
    model = MLP(hidden_dim, **kw_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    hr10 = []
    
    print(f'======================== new run - {recommender_name} ========================')
    logger.info(f'======================== new run - {recommender_name} ========================')
    
    num_training = train_data.shape[0]
    num_batches = int(np.ceil(num_training / batch_size))

    
    for epoch in range(epochs):
        train_matrix = sample_indices(train_data.copy(), **kw_dict)
        perm = np.random.permutation(num_training)
        loss = []
        train_pos_loss=[]
        train_neg_loss=[]
        if epoch!=0 and epoch%10 == 0: # decrease the learning rate every 10 epochs
            lr = 0.1*lr
            optimizer.lr = lr
        
        for b in range(num_batches):
            optimizer.zero_grad()
            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]    
            batch_matrix = torch.FloatTensor(train_matrix[batch_idx,:-2]).to(device) #refers to all columns up to but not including the second-to-last column

            batch_pos_idx = train_matrix[batch_idx,-2] # refers to the second-to-last column in these rows
            batch_neg_idx = train_matrix[batch_idx,-1] # refers to the last column in these rows
            
            batch_pos_items = torch.Tensor(items_array[batch_pos_idx]).to(device)# [256,3381]
            batch_neg_items = torch.Tensor(items_array[batch_neg_idx]).to(device)# [256,3381]
            
            pos_output = torch.diagonal(model(batch_matrix, batch_pos_items))  # [256]
            neg_output = torch.diagonal(model(batch_matrix, batch_neg_items)) # [256]
            
            # MSE loss MSE = (1/N) * sum((y-y_pred)^2)
            pos_loss = torch.mean((torch.ones_like(pos_output)-pos_output)**2)
            neg_loss = torch.mean((neg_output)**2)
            
            batch_loss = pos_loss + beta*neg_loss
            batch_loss.backward()
            optimizer.step()
            
            loss.append(batch_loss.item())
            train_pos_loss.append(pos_loss.item())
            train_neg_loss.append(neg_loss.item())
            
        print(f'train pos_loss = {np.mean(train_pos_loss)}, neg_loss = {np.mean(train_neg_loss)}')    
        train_losses.append(np.mean(loss))
        if epoch % ((epochs*5)/100) == 0:
            print(f'this model is saved as MLP_{data_name}_{epoch}_{hidden_dim}.pt')
            torch.save(model.state_dict(), Path(checkpoints_path, f'MLP_{data_name}_{epoch}_{hidden_dim}.pt'))


        model.eval()
        test_matrix = np.array(static_test_data)
        test_tensor = torch.Tensor(test_matrix[:,:-2]).to(device)
        
        test_pos = test_matrix[:,-2] # [256]
        test_neg = test_matrix[:,-1]
        
        row_indices = np.arange(test_matrix.shape[0]) # [256]
        test_tensor[row_indices,test_pos] = 0
        
        pos_items = torch.Tensor(items_array[test_pos]).to(device) # [256,3381]
        neg_items = torch.Tensor(items_array[test_neg]).to(device)
        
        pos_output = torch.diagonal(model(test_tensor, pos_items).to(device))
        neg_output = torch.diagonal(model(test_tensor, neg_items).to(device))
        
        pos_loss = torch.mean((torch.ones_like(pos_output)-pos_output)**2)
        neg_loss = torch.mean((neg_output)**2)
        print(f'test pos_loss = {pos_loss}, neg_loss = {neg_loss}')
        
        hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model,batch_index=batch_idx, **kw_dict)
        hr10.append(hit_rate_at_10) # metric for monitoring
        print(hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR)
        
        test_losses.append(-hit_rate_at_10)
        if epoch>5: # early stop if the HR@10 decreases for 4 epochs in a row
            if test_losses[-2]<=test_losses[-1] and test_losses[-3]<=test_losses[-2] and test_losses[-4]<=test_losses[-3]:
                logger.info(f'Early stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
                train_losses_dict[trial.number] = train_losses
                test_losses_dict[trial.number] = test_losses
                HR10_dict[trial.number] = hr10
                return max(hr10)
            
    logger.info(f'Stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
    train_losses_dict[trial.number] = train_losses
    test_losses_dict[trial.number] = test_losses
    HR10_dict[trial.number] = hr10
    return max(hr10)

VAE_config= {
"enc_dims": [512,64],#change 256 to 512
"dropout": 0.5,
"anneal_cap": 0.2,
"total_anneal_steps": 200000
}

def VAE_objective(trial):
    
    lr = trial.suggest_float('learning_rate', 0.001, 0.01)
    batch_size = trial.suggest_categorical('batch_size', [64,128,256])
    epochs = 20
    model = VAE(VAE_config ,**kw_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    hr10 = []
    print('======================== new run ========================')
    logger.info('======================== new run ========================')
    
    for epoch in range(epochs):
        if epoch!=0 and epoch%10 == 0:
            lr = 0.1*lr
            optimizer.lr = lr
        loss = model.train_one_epoch(train_array, optimizer, batch_size)
        train_losses.append(loss)


        model.eval()
        test_matrix = static_test_data.to_numpy()
        test_tensor = torch.Tensor(test_matrix[:,:-2]).to(device)
        test_pos = test_array[:,-2]
        test_neg = test_array[:,-1]
        row_indices = np.arange(test_matrix.shape[0])
        test_tensor[row_indices,test_pos] = 0
        output = model(test_tensor).to(device)
        pos_loss = -output[row_indices,test_pos].mean()
        neg_loss = output[row_indices,test_neg].mean()
        print(f'pos_loss = {pos_loss}, neg_loss = {neg_loss}')
        
        hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, **kw_dict)
        hr10.append(hit_rate_at_10)
        print(hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR)
        if epoch % ((epochs*5)/100) == 0:
            print(f'this model is saved as VAE_{data_name}_{epoch}.pt')
            torch.save(model.state_dict(), Path(checkpoints_path, f'VAE_{data_name}_{epoch}.pt'))

        
        test_losses.append(pos_loss.item())
        if epoch>5:
            if test_losses[-2]<test_losses[-1] and test_losses[-3]<test_losses[-2] and test_losses[-4]<test_losses[-3]:
                logger.info(f'Early stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
                train_losses_dict[trial.number] = train_losses
                test_losses_dict[trial.number] = test_losses
                HR10_dict[trial.number] = hr10
                return max(hr10)
    
    logger.info(f'Stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
    train_losses_dict[trial.number] = train_losses
    test_losses_dict[trial.number] = test_losses
    HR10_dict[trial.number] = hr10
    return max(hr10)


# ------------------ LightGCN ------------------
def load_edge_csv(df,
                #   src_index_col,
                #   dst_index_col,
                  link_index_col=None,
                  rating_treshold=3):
    edge_index = None
    src = [user_id for user_id in df['userId']]
    dst = [movie_id for movie_id in df['movieId']]

    edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) >= rating_treshold

    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])
    return edge_index

def convert_r_mat_edge_index_to_adj_mat_edge_index(input_edge_index, num_users, num_items):
    R = torch.zeros((num_users, num_items))
    
    # convert sparse coo format to dense format to get R matrix
    for i in range(len(input_edge_index[0])):
        row_idx = input_edge_index[0][i]
        col_idx = input_edge_index[1][i]
        R[row_idx][col_idx] = 1

    R_transpose = torch.transpose(R, 0, 1)
    adj_mat = torch.zeros((num_users + num_items, num_users + num_items))
    adj_mat[:num_users, num_users:] = R.clone()
    adj_mat[num_users:, :num_users] = R_transpose.clone()
    adj_mat_coo = adj_mat.to_sparse_coo()
    adj_mat_coo = adj_mat_coo.indices()
    return adj_mat_coo

def convert_adj_mat_edge_index_to_r_mat_edge_index(input_edge_index):
    sparse_input_edge_index = SparseTensor(row=input_edge_index[0],
                                            col=input_edge_index[1],
                                            sparse_sizes=(num_users + num_items, num_users + num_items))
    adj_mat = sparse_input_edge_index.to_dense()
    interact_mat = adj_mat[:num_users, num_users:] 
    r_mat_edge_index = interact_mat.to_sparse_coo().indices()
    return r_mat_edge_index

def sample_mini_batch(batch_size, edge_index):
    edges = structured_negative_sampling(edge_index, num_nodes=num_items)
    edges = torch.stack(edges, dim=0)
    indices = random.choices([i for i in range(edges.size(0))], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices

# ## Data imports and preprocessing
def create_edge_index(X):
    edge_index = []
    for user_id in X.index:
        for item_id in X.columns:
            if X.loc[user_id, item_id] > 0:  # Check if there is an interaction
                edge_index.append([user_id, int(item_id)])
    
    edge_index = np.array(edge_index).T
    return edge_index



output_type = output_type_dict[recommender_name] ### Can be single, multiple
num_users = num_users_dict[data_name] 
num_items = num_items_dict[data_name] 

train_data = pd.read_csv(Path(files_path,f'train_data_{data_name}.csv'), index_col=0)
test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
static_test_data = pd.read_csv(Path(files_path,f'static_test_data_{data_name}.csv'), index_col=0)
with open(Path(files_path,f'pop_dict_{data_name}.pkl'), 'rb') as f:
    pop_dict = pickle.load(f)
train_array = train_data.to_numpy()
test_array = test_data.to_numpy()
items_array = np.eye(num_items)
all_items_tensor = torch.Tensor(items_array).to(device)


for row in range(static_test_data.shape[0]):
    static_test_data.iloc[row, static_test_data.iloc[row,-2]]=0
test_array = static_test_data.iloc[:,:-2].to_numpy()

# static_test_data

pop_array = np.zeros(len(pop_dict))
for key, value in pop_dict.items():
    pop_array[key] = value

# edge_index = convert_R_to_adj_mat_edge_index(train_data, num_users, num_items)


    #----create edge_index for LightGCN
if False:
    edge_index_train = create_edge_index(train_data)
    edge_index_test = create_edge_index(test_data)
    np.save(Path(files_path, f'edge_index_train_{data_name}.npy'), edge_index_train)
    np.save(Path(files_path, f'edge_index_test_{data_name}.npy'), edge_index_test)
    edge_index_train = torch.LongTensor(edge_index_train).to(device)
    edge_index_test = torch.LongTensor(edge_index_test).to(device)
    train_edge_index = edge_index_train
    num_interactions = edge_index_test.shape[1]

    # split the edges of the graphs using a 80/10/10 ratio
    all_indices = [i for i in range(num_interactions)]
    test_indices, val_indices = train_test_split(all_indices, test_size=0.5, random_state=1)
    val_edge_index = edge_index_test[:, val_indices]
    test_edge_index = edge_index_test[:, test_indices]
else:
    edge_index_train = np.load(Path(files_path, f'edge_index_train_{data_name}.npy'))
    edge_index_train = torch.LongTensor(edge_index_train).to(device)
    edge_index_test = np.load(Path(files_path, f'edge_index_test_{data_name}.npy'))
    edge_index_test = torch.LongTensor(edge_index_test).to(device)
    # val_edge_index, test_edge_index = train_test_split(edge_index_test, test_size=0.5, random_state=1)
    train_edge_index = edge_index_train

    num_interactions = edge_index_test.shape[1]

    # split the edges of the graphs using a 80/10/10 ratio
    all_indices = [i for i in range(num_interactions)]
    test_indices, val_indices = train_test_split(all_indices, test_size=0.5, random_state=1)
    val_edge_index = edge_index_test[:, val_indices]
    test_edge_index = edge_index_test[:, test_indices]

edge_index = pd.concat([train_data, test_data])
r_mat_edge_index = sparse.coo_matrix(edge_index.to_numpy())
r_mat_edge_index = [r_mat_edge_index.row, r_mat_edge_index.col]
# edge_index = torch.LongTensor(r_mat_edge_index)
r_mat_edge_index_np = np.array(r_mat_edge_index)
# Convert the numpy array to a PyTorch tensor
edge_index = torch.LongTensor(r_mat_edge_index_np)
# edge_index = convert_r_mat_edge_index_to_adj_mat_edge_index(edge_index, num_users, num_items)
edge_index = edge_index.to(device)


    # convert from r_mat edge index to adj matirices edge index
    # so we can feed it to the GCN model

train_edge_index = convert_r_mat_edge_index_to_adj_mat_edge_index(train_edge_index, num_users, num_items)
val_edge_index = convert_r_mat_edge_index_to_adj_mat_edge_index(val_edge_index, num_users, num_items)
test_edge_index = convert_r_mat_edge_index_to_adj_mat_edge_index(test_edge_index, num_users, num_items)

print(edge_index_train)
print(edge_index_train.size())
print(val_edge_index)
print(val_edge_index.size())
print(test_edge_index)
print(test_edge_index.size())
"""
# before we apply rating treshhold
# num_users = len(rating_df['userId'].unique())
# num_movies = len(rating_df['movieId'].unique())

# num_interactions = edge_index.shape[1]

# split the edges of the graphs using a 80/10/10 ratio
# all_indices = [i for i in range(num_interactions)]

# train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=1)

# val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=1)

# train_edge_index = edge_index[:, train_indices]
# val_edge_index = edge_index[:, val_indices]
# test_edge_index = edge_index[:, test_indices]

# print(f"num_users: {num_users}, num_movies: {num_movies}, num_interactions: {num_interactions}")
# print(f"train_edge_index: {train_edge_index.size()}")
# print(f"num_users + num_movies: {num_users + num_movies}")
# print(torch.unique(train_edge_index[0]).size())
# print(torch.unique(train_edge_index[1]).size())


# convert from r_mat edge index to adj matirices edge index
# so we can feed it to the GCN model
# train_edge_index = convert_r_mat_edge_index_to_adj_mat_edge_index(train_edge_index, num_users, num_movies)
# val_edge_index = convert_r_mat_edge_index_to_adj_mat_edge_index(val_edge_index, num_users, num_movies)
# test_edge_index = convert_r_mat_edge_index_to_adj_mat_edge_index(test_edge_index, num_users, num_movies)

# print(train_edge_index)
# print(train_edge_index.size())
# print(val_edge_index)
# print(val_edge_index.size())
# print(test_edge_index)
# print(test_edge_index.size())
"""

class LightGCN(MessagePassing):
    def __init__(self, num_users,
                 num_items,
                 embedding_dim=64,
                 K=3,
                 add_self_loops=False):
        super().__init__()
        self.num_users = num_users
        # self.num_items = num_items
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K = K
        self.add_self_loops = add_self_loops

        self.user_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.item_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, edge_index: Tensor):
        edge_index_norm = gcn_norm(edge_index= edge_index, add_self_loops=self.add_self_loops)
        emb_0 = torch.cat([self.user_emb.weight, self.item_emb.weight])
        embs = [emb_0]
        emb_k = emb_0
        for k in range(self.K):
            emb_k = self.propagate(edge_index=edge_index_norm[0], x=emb_k, norm=edge_index_norm[1])
            embs.append(emb_k)
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)
        user_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])
        return user_emb_final, self.user_emb.weight, items_emb_final, self.item_emb.weight
    def message(self, x_j: Tensor, norm: Tensor):
        return norm.view(-1, 1) * x_j
    
def bpr_loss(users_emb_final,
             users_emb_0,
             pos_items_emb_final,
             pos_items_emb_0,
             neg_items_emb_final,
             neg_items_emb_0,
             lambda_val):
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) + 
                             pos_items_emb_0.norm(2).pow(2) + 
                             neg_items_emb_0.norm(2).pow(2))
    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)

    bpr_loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores))
    loss = bpr_loss + reg_loss
    return loss

def get_user_positive_items(edge_index):
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user_id = edge_index[0][i].item()
        item_id = edge_index[1][i].item()
        if user_id not in user_pos_items:
            user_pos_items[user_id] = []
        user_pos_items[user_id].append(item_id)
    return user_pos_items

def RecallPrecision_ATK(groungTruth, r, k):
    num_correct_pred = torch.sum(r, dim=-1)
    user_num_liked = torch.Tensor([len(groungTruth[i]) for i in range(len(groungTruth))])
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred / k)
    return recall.item(), precision.item()

# computes NDCG@K
def NDCGatK_r(groungTruth, r, k):
    assert len(r) == len(groungTruth)
    test_matrix = torch.zeros((len(r),k))
    for i, items in enumerate(groungTruth):
        length = min(len(items), k)
        test_matrix[i,:length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1 / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1 / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0] = 1
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0
    return torch.mean(ndcg).item()

# wraper function to getevaluation metrics
def get_metrics(model,
                input_edge_index,
                input_exclude_edge_indices,
                k):
    user_embedding = model.user_emb.weight
    item_embedding = model.item_emb.weight

    edge_index = convert_adj_mat_edge_index_to_r_mat_edge_index(input_edge_index)

    exclude_edge_indices = [convert_adj_mat_edge_index_to_r_mat_edge_index(exclude_edge_index) \
                            for exclude_edge_index in input_exclude_edge_indices]
    
    r_mat_rating = torch.matmul(user_embedding, item_embedding.T)
    rating = r_mat_rating 

    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_user_positive_items(exclude_edge_index)
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        rating[exclude_users, exclude_items] = -(1 << 10)
    _, top_k_items = torch.topk(rating, k=k)
    users = edge_index[0].unique()
    test_user_pos_items = get_user_positive_items(edge_index)
    test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]
    r = []
    for user in users:
        user_true_relevant_item = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in user_true_relevant_item, top_k_items[user]))
        r.append(label)

    r = torch.Tensor(np.array(r).astype('float'))
    recall, precision = RecallPrecision_ATK(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)

    return recall, precision, ndcg

# wraper function for evaluation model
def evaluation(model,
               edge_index,
               exclude_edge_indices,
               k,
               lambda_val):
    user_emb_final, user_emb_0, items_emb_final, items_emb_0 = model.forward(edge_index)
    r_mat_edge_index = convert_adj_mat_edge_index_to_r_mat_edge_index(edge_index)
    edges = structured_negative_sampling(r_mat_edge_index, contains_neg_self_loops=False, num_nodes=num_items)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    users_emb_final, users_emb_0 = user_emb_final[user_indices], user_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]
    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, lambda_val)
    recall, precision, ndcg = get_metrics(model, edge_index, exclude_edge_indices, k)
    return loss, recall, precision, ndcg

def get_embs_for_bpr(model, input_edge_index):
    user_emb_final, user_emb_0, items_emb_final, items_emb_0 = model.forward(input_edge_index)
    edge_index_to_use = convert_adj_mat_edge_index_to_r_mat_edge_index(input_edge_index)
    user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(BATCH_SIZE, edge_index_to_use)
    user_indices, pos_item_indices, neg_item_indices = user_indices.to(device), pos_item_indices.to(device), neg_item_indices.to(device)

    users_emb_final, users_emb_0 = user_emb_final[user_indices], user_emb_0[user_indices]
    pos_item_emb_final, pos_item_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_item_emb_final, neg_item_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]

    return users_emb_final, users_emb_0, pos_item_emb_final, pos_item_emb_0, neg_item_emb_final, neg_item_emb_0

def LightGCN_objective(trial):
    layers = 3
    # model = LightGCN(num_users=user_ids.shape[0], num_items=item_ids.shape[0], embedding_dim=64, K=layers)
    model = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=64, K=layers)


    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    # print(f"Device: {device}")

    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=ITERS_PER_LR_DECAY, gamma=0.95)

    global edge_index, train_edge_index, val_edge_index
    edge_index = edge_index.to(device) # shape is [2, 48580]
    train_edge_index = train_edge_index.to(device) # shape is [2, 77728]
    val_edge_index = val_edge_index.to(device) # shape is [2, 9716]

    # training loop
    train_losses = []
    val_losses = []
    val_recall_at_ks = []

    for iter in tqdm(range(ITERATIONS)):
        # forward pass
        users_emb_final, users_emb_0, pos_item_emb_final, pos_item_emb_0, neg_item_emb_final, neg_item_emb_0 = get_embs_for_bpr(model, train_edge_index)

        # loss computation
        train_loss = bpr_loss(users_emb_final, users_emb_0, pos_item_emb_final, pos_item_emb_0, neg_item_emb_final, neg_item_emb_0, LAMBDA)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # validation
        if iter % ITERS_PER_EVAL == 0:
            model.eval()


            with torch.no_grad():
                val_loss, val_recall, val_precision, val_ndcg = evaluation(model, val_edge_index, [train_edge_index], 10, LAMBDA)
                val_losses.append(val_loss)
                val_recall_at_ks.append(val_recall)
                print(f"Validation loss: {val_loss}, Recall@10: {val_recall}, Precision@10: {val_precision}, NDCG@10: {val_ndcg}")

                train_losses.append(train_loss.item())
                val_losses.append(val_loss)
                val_recall_at_ks.append(round(val_recall, 5))
            model.train()
        if iter % ITERS_PER_LR_DECAY == 0 and iter > 0:
            scheduler.step()
    torch.save(model.state_dict(), Path(checkpoints_path, f'LightGCN_{data_name}_New.pt'))
    return True


def load_recommender():
    if recommender_name=='MLP':
        recommender = MLP(hidden_dim, **kw_dict)
    elif recommender_name=='VAE':
        recommender = VAE(VAE_config, **kw_dict)
    elif recommender_name=='LightGCN':
        recommender = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=64, K=3)
    recommender_checkpoint = torch.load(Path(checkpoints_path, recommender_path)) #map_location=torch.device('cpu')
    recommender.load_state_dict(recommender_checkpoint)
    recommender.eval()
    for param in recommender.parameters():
        param.requires_grad= False
    return recommender



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="List files in a directory that start with a given keyword.")
    
    # Add the arguments
    parser.add_argument('directory', type=str, default="MLP1_ML1M_0.0076_256_7.pt", nargs='?')
    parser.add_argument('model', type=str, default="MLP", nargs='?')
    
    # Parse the arguments
    args = parser.parse_args()
    recommender_name = args.model

    # # Help functions

    kw_dict = {'device':device,
            'num_items': num_items,
            'pop_array':pop_array,
            'all_items_tensor':all_items_tensor,
            'static_test_data':static_test_data,
            'items_array':items_array,
            'output_type':output_type,
            'recommender_name':recommender_name,
            'edge_index':edge_index_train}

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler(f"{recommender_name}_{data_name}_Optuna.log", mode="w")) #TODO change folder

    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    study = optuna.create_study(direction='maximize')

    logger.info("Start optimization.")

    if recommender_name == 'MLP':
        study.optimize(MLP_objective, n_trials=3) 
    elif recommender_name == 'VAE':
        study.optimize(VAE_objective, n_trials=1) 
    elif recommender_name == 'LightGCN':
        study.optimize(LightGCN_objective, n_trials=1)

    with open(f"{recommender_name}_{data_name}_Optuna.log") as f:
        assert f.readline().startswith("A new study created")
        assert f.readline() == "Start optimization.\n"
        
        
    # Print best hyperparameters and corresponding metric value
    print("Best hyperparameters: {}".format(study.best_params))
    print("Best metric value: {}".format(study.best_value))

  
    recommender_path_dict = {
        ("ML1M","VAE"): Path(checkpoints_path, "VAE_ML1M_0.0007_128_10.pt"),
        ("ML1M","MLP"):Path(checkpoints_path, "MLP1_ML1M_0.0076_256_7.pt"),
        ("ML1M","LightGCN"):Path(checkpoints_path, "LightGCN_ML1M_New.pt"),

        ("Yahoo","VAE"): Path(checkpoints_path, "VAE_Yahoo_0.0001_128_13.pt"),
        ("Yahoo","MLP"):Path(checkpoints_path, "MLP2_Yahoo_0.0083_128_1.pt"),
        ("Yahoo","LightGCN"):Path(checkpoints_path, "LightGCN_Yahoo_.pt"),
        
        ("Pinterest","VAE"): Path(checkpoints_path, "VAE_Pinterest_12_18_0.0001_256.pt"),
        ("Pinterest","MLP"):Path(checkpoints_path, "MLP_Pinterest_0.0062_512_21_0.pt"),
        ("Pinterest","LightGCN"):Path(checkpoints_path, "LightGCN_Pinterest_.pt"),
        
    }

    hidden_dim_dict = {
        ("ML1M","VAE"): None,
        ("ML1M","MLP"): 32,
        ("ML1M","LightGCN"): 64,

        ("Yahoo","VAE"): None,
        ("Yahoo","MLP"):32,
        ("Yahoo","LightGCN"):64,
        
        ("Pinterest","VAE"): None,
        ("Pinterest","MLP"):512,
        ("Pinterest","LightGCN"):64,
    }

    hidden_dim = hidden_dim_dict[(data_name,recommender_name)]
    recommender_path = recommender_path_dict[(data_name,recommender_name)]


    
    # model = load_recommender()

    # ### Calculate recommender's HR@10, HR@50, HR@100, MRR and MPR

    # hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, **kw_dict)

    # print(f'hit_rate_at_10 is {hit_rate_at_10}, hit_rate_at_50 is {hit_rate_at_50}, hit_rate_at_100 is {hit_rate_at_100},  MRR: {MRR}, MPR:{MPR}')

# python code/LXR_training.py --directory MLP_ML1M_1_512.pt --model MLP
#sbatch --job-name=CFX1 --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/LXR_training.py --directory MLP_ML1M_1_512.pt --model MLP"
from contextlib import contextmanager
from time import time, sleep

@contextmanager
def timer():
    start_time = time()
    yield
    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")

# # Imports

import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
export_dir = os.getcwd()
from pathlib import Path
import pickle
from collections import defaultdict
import time
import torch
import torch.nn as nn
import copy
import optuna
import logging
import matplotlib.pyplot as plt
import random
import ipynb
import wandb
import importlib

import os,argparse
parser = argparse.ArgumentParser(description="List files in a directory that start with a given keyword.")

# Add the arguments
parser.add_argument('--directory', type=str, default="VAE_ML1M_17.pt", nargs='?')
parser.add_argument('--model', type=str, default="VAE", nargs='?')


# Parse the arguments
args = parser.parse_args()
recommender_name = args.model




data_name = "ML1M" ### Can be ML1M, Yahoo, Pinterest
recommender_name = args.model ## Can be MLP, VAE, LightGCN
k = 10
print(f'method {recommender_name} @ {k}')
n_trials = 1
DP_DIR = Path("processed_data", data_name) 
export_dir = Path(os.getcwd())
files_path = Path(export_dir, "processed_data", data_name)
checkpoints_path = Path(export_dir, "checkpoints")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu' #TODO

# LightGCN -----------------------------------------------------
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
    pos_scores = torch.sum(pos_scores, dim=1)
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=1)

    bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).sum()
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
    # For each user-item pair that should be excluded (e.g., items the user has already interacted with in training), set the rating to a very low value to prevent them from being recommended.
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=ITERS_PER_LR_DECAY, gamma=0.5)

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
    torch.save(model.state_dict(), Path(checkpoints_path, f'LightGCN_{data_name}_iter10k2.pt'))
    return True


# finish LightGCN -----------------------------------------------------


output_type_dict = {
    "VAE":"multiple",
    "MLP":"single",
    "LightGCN":"single"
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



recommender_path_dict = {
    # ("ML1M","VAE"): Path(checkpoints_path, "VAE_ML1M_0.0007_128_10.pt"),
    ("ML1M","VAE"): Path(checkpoints_path, args.directory),
    ("ML1M","MLP"):Path(checkpoints_path, args.directory),
    ("ML1M","LightGCN"):Path(checkpoints_path, "LightGCN_ML1M_New.pt"),
    
    ("Yahoo","VAE"): Path(checkpoints_path, "VAE_Yahoo_0.0001_128_13.pt"),
    ("Yahoo","MLP"):Path(checkpoints_path, "MLP2_Yahoo_0.0083_128_1.pt"),

    ("Pinterest","VAE"): Path(checkpoints_path, "VAE_Pinterest_0.0002_32_12.pt"),
    ("Pinterest","MLP"):Path(checkpoints_path, "MLP_Pinterest_0.0062_512_21_0.pt")
    
}

hidden_dim_dict = {
    ("ML1M","VAE"): None,
    ("ML1M","MLP"): 512, #change from 32
    ("ML1M","LightGCN"): 64,

    ("Yahoo","VAE"): None,
    ("Yahoo","MLP"):32,
    #("Yahoo","LightGCN"): 64,
    
    ("Pinterest","VAE"): None,
    ("Pinterest","MLP"):512
    #("Pinterest","LightGCN"): 64

}

output_type = output_type_dict[recommender_name] ### Can be single, multiple
num_users = num_users_dict[data_name] 
num_items = num_items_dict[data_name] 

hidden_dim = hidden_dim_dict[(data_name,recommender_name)]
recommender_path = recommender_path_dict[(data_name,recommender_name)]

# ## Data imports and preprocessing

train_data = pd.read_csv(Path(files_path,f'train_data_{data_name}.csv'), index_col=0)
test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
train_data['user_id'] = train_data.index
test_data['user_id'] = test_data.index
static_test_data = pd.read_csv(Path(files_path,f'static_test_data_{data_name}.csv'), index_col=0)
with open(Path(files_path,f'pop_dict_{data_name}.pkl'), 'rb') as f:
    pop_dict = pickle.load(f)
train_array = train_data.to_numpy()
test_array = test_data.to_numpy()
items_array = np.eye(num_items)
all_items_tensor = torch.Tensor(items_array).to(device)

pop_array = np.zeros(len(pop_dict))
for key, value in pop_dict.items():
    pop_array[key] = value

# create edge_index for LightGCN
# if recommender_name == 'LightGCN':
edge_index_train = np.load(Path(files_path, f'edge_index_train_{data_name}.npy'))
edge_index_train = torch.LongTensor(edge_index_train).to(device)
edge_index_test = np.load(Path(files_path, f'edge_index_test_{data_name}.npy'))
edge_index_test = torch.LongTensor(edge_index_test).to(device)
edge_index = pd.concat([train_data, test_data])
r_mat_edge_index = sparse.coo_matrix(edge_index.to_numpy())
r_mat_edge_index = np.array([r_mat_edge_index.row, r_mat_edge_index.col])
edge_index = torch.LongTensor(r_mat_edge_index).to(device)


kw_dict = {'device':device,
          'num_items': num_items,
          'pop_array':pop_array,
          'all_items_tensor':all_items_tensor,
          'static_test_data':static_test_data,
          'items_array':items_array,
          'output_type':output_type,
          'recommender_name':recommender_name,
          'edge_index':edge_index
          }

# # Recommenders Architecture

from recommenders_architecture import *
# from recommenders_training import LightGCN

VAE_config= {
"enc_dims": [512,64],
"dropout": 0.5,
"anneal_cap": 0.2,
"total_anneal_steps": 200000
}

def load_recommender():
    if recommender_name=='MLP':
        recommender = MLP(hidden_dim, **kw_dict)
    elif recommender_name=='VAE':
        recommender = VAE(VAE_config, **kw_dict)
    elif recommender_name=='LightGCN':
        recommender = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=64, K=3)

    recommender_checkpoint = torch.load(Path(checkpoints_path, recommender_path), map_location=torch.device('cpu')) #, map_location=torch.device('cpu')
    recommender.load_state_dict(recommender_checkpoint)
    recommender.eval()
    for param in recommender.parameters():
        param.requires_grad= False
    return recommender
    
recommender = load_recommender()

# # Help functions
from help_functions import *

# ## Load / create top recommended items dict

create_dicts = True
if create_dicts:
    top1_train = {}
    top1_test = {}
    for i in range(train_array.shape[0]):
        user_index = train_array[i][-1]
        user_tensor = torch.Tensor(train_array[i][:-1]).to(device)
        top1_train[user_index] = int(get_user_recommended_item(user_tensor, recommender, user_index, **kw_dict))
    for i in range(test_array.shape[0]):
        user_index = test_array[i][-1]
        user_tensor = torch.Tensor(test_array[i][:-1]).to(device)
        top1_test[user_index] = int(get_user_recommended_item(user_tensor, recommender, user_index, **kw_dict))
        
    with open(Path(files_path,f'top1_train_{data_name}_{recommender_name}_New.pkl'), 'wb') as f:
        pickle.dump(top1_train, f)
    with open(Path(files_path,f'top1_test_{data_name}_{recommender_name}_New.pkl'), 'wb') as f:
        pickle.dump(top1_test, f)
else:
    with open(Path(files_path,f'top1_train_{data_name}_{recommender_name}.pkl'), 'rb') as f:
        top1_train = pickle.load(f)
    with open(Path(files_path,f'top1_test_{data_name}_{recommender_name}.pkl'), 'rb') as f:
        top1_test = pickle.load(f)

# # LXR Architecture

class Explainer(nn.Module):
    def __init__(self, user_size, item_size, hidden_size):
        super(Explainer, self).__init__()
        
        self.users_fc = nn.Linear(in_features = user_size, out_features=hidden_size).to(device)
        self.items_fc = nn.Linear(in_features = item_size, out_features=hidden_size).to(device)
        self.bottleneck = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features = hidden_size*2, out_features=hidden_size).to(device),
            nn.Tanh(),
            nn.Linear(in_features = hidden_size, out_features=user_size).to(device),
            nn.Sigmoid()
        ).to(device)
        
        
    def forward(self, user_tensor, item_tensor):
        user_output = self.users_fc(user_tensor.float())
        item_output = self.items_fc(item_tensor.float())
        combined_output = torch.cat((user_output, item_output), dim=-1)
        expl_scores = self.bottleneck(combined_output).to(device)
        return expl_scores

# # Train functions

class LXR_loss(nn.Module):
    def __init__(self, lambda_pos, lambda_neg, alpha):
        super(LXR_loss, self).__init__()
        
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.alpha = alpha
          
    def forward(self, user_tensors, items_tensors, items_ids, pos_masks):
        neg_masks = torch.sub(torch.ones_like(pos_masks), pos_masks)
        x_masked_pos = user_tensors * pos_masks
        x_masked_neg = user_tensors * neg_masks
        if output_type=='single':
            x_masked_res_pos = torch.diag(recommender_run(x_masked_pos, recommender, items_tensors, item_id=items_ids, wanted_output = 'single', **kw_dict))
            x_masked_res_neg = torch.diag(recommender_run(x_masked_neg, recommender, items_tensors, item_id=items_ids, wanted_output = 'single', **kw_dict))
        else:
            x_masked_res_pos_before = recommender_run(x_masked_pos, recommender, items_tensors, item_id=items_ids, wanted_output = 'vector', **kw_dict)
            x_masked_res_neg_before = recommender_run(x_masked_neg, recommender, items_tensors, item_id=items_ids, wanted_output = 'vector', **kw_dict)
            rows=torch.arange(len(items_ids))
            x_masked_res_pos = x_masked_res_pos_before[rows, items_ids] 
            x_masked_res_neg = x_masked_res_neg_before[rows, items_ids] 
            
            
        pos_loss = -torch.mean(torch.log(x_masked_res_pos))
        neg_loss = torch.mean(torch.log(x_masked_res_neg))
        l1 = x_masked_pos[user_tensors>0].mean()
        combined_loss = self.lambda_pos*pos_loss + self.lambda_neg*neg_loss + self.alpha*l1
        
        return combined_loss, pos_loss, neg_loss, l1

#LXR based similarity
def find_LXR_mask(user_tensor, item_id, item_tensor, explainer):
    expl_scores = explainer(user_tensor, item_tensor)
    x_masked = user_tensor*expl_scores
    item_sim_dict = {i: x_masked[i].item() for i in range(len(x_masked))}    

    return item_sim_dict

# evaluate LXR pos@k and neg@k scores on test set
def calculate_pos_neg_k(user_tensor, item_id, items_tensor, num_of_bins, explainer, k):
    POS_masked = user_tensor
    NEG_masked = user_tensor
    user_hist_size = int(torch.sum(user_tensor))

    bins = [0] + [len(x) for x in np.array_split(np.arange(user_hist_size), num_of_bins, axis=0)]

    POS_at_k = [0] * (num_of_bins+1)
    NEG_at_k = [0] * (num_of_bins+1)
    total_items = 0
    
    #returns original tensor
    sim_items = find_LXR_mask(user_tensor, item_id, items_tensor, explainer)
    POS_sim_items=list(sorted(sim_items.items(), key=lambda item: item[1],reverse=True))[0:user_hist_size]
    NEG_sim_items  = list(sorted(dict(POS_sim_items).items(), key=lambda item: item[1],reverse=False))
    
    for i in range(len(bins)):
        total_items += bins[i]
        
        POS_masked = torch.zeros_like(user_tensor, dtype=torch.float32, device=device)
        for j in POS_sim_items[:total_items]:
            POS_masked[j[0]] = 1
        POS_masked = user_tensor - POS_masked # remove the masked items from the user history 
        
        NEG_masked = torch.zeros_like(user_tensor, dtype=torch.float32, device=device)
        for j in NEG_sim_items[:total_items]:
            NEG_masked[j[0]] = 1
        NEG_masked = user_tensor - NEG_masked # remove the masked items from the user history 
        
        POS_index = get_index_in_the_list(POS_masked, user_tensor, item_id, recommender, **kw_dict)+1
        NEG_index = get_index_in_the_list(NEG_masked, user_tensor, item_id, recommender, **kw_dict)+1        
        
        POS_at_k[i] = 1 if POS_index <= k else 0 #change
        NEG_at_k[i] = 1 if NEG_index <= k else 0 #change

    res = [np.array(POS_at_k), np.array(NEG_at_k)]
    return res

# # LXR training
# ### Utilizing Optuna for hyperparameter optimization and WandB (Weights and Biases) for experiment tracking and logging.

torch.manual_seed(42)
np.random.seed(42)

num_of_rand_users = 200 # number of users for evaluations
random_rows = np.random.choice(test_array.shape[0], num_of_rand_users, replace=False)
random_sampled_array = test_array[random_rows]

def lxr_training(trial):
    print(f'model is {args.directory}')
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01)
    alpha = trial.suggest_categorical('alpha', [1]) # set alpha to be 1, change other hyperparameters
    lambda_neg = trial.suggest_float('lambda_neg', 0,50)
    lambda_pos = trial.suggest_float('lambda_pos', 0,50)
    batch_size = trial.suggest_categorical('batch_size', [32,64,128,256])
    explainer_hidden_size = trial.suggest_categorical('explainer_hidden_size', [32,64,128])
    epochs = 40 #change
    
    wandb.init(
        project=f"{data_name}_{recommender_name}_LXR_training",
        name=f"trial_{trial.number}",
        config={
        'learning_rate' : learning_rate,
        'alpha' : alpha,
        'lambda_neg' : lambda_neg,
        'lambda_pos' : lambda_pos,
        'batch_size' : batch_size,
        'explainer_hidden_size' : explainer_hidden_size,
        'architecture' : 'LXR_combined',
        'activation_function' : 'Tanh',
        'loss_type' : 'logloss',
        'optimize_for' : 'pos_at_k',
        'epochs':epochs
        })

    loader = torch.utils.data.DataLoader(train_array, batch_size=batch_size, shuffle=True)
    num_batches = int(np.ceil(train_array.shape[0] / batch_size))


    num_of_bins = 10
    run_pos_at_k = []
    run_neg_at_k = []
    metric_for_monitoring = []
    train_losses = []

    recommender.eval()
    explainer = Explainer(train_data.shape[1]-1, num_items, explainer_hidden_size).to(device)  # changed num_features to num_users #TODO 
    optimizer_comb = torch.optim.Adam(explainer.parameters(), learning_rate)
    loss_func = LXR_loss(lambda_pos, lambda_neg, alpha)

    print('======================== new run ========================')

    for epoch in range(epochs):
        if epoch%15 == 0 and epoch>0: # decrease learning rate every 15 epochs
            learning_rate*= 0.1
            optimizer_comb.lr = learning_rate

        POS_at_k_lxr = np.zeros(num_of_bins+1)
        NEG_at_k_lxr = np.zeros(num_of_bins+1)
        train_loss = 0
        total_pos_loss=0
        total_neg_loss=0
        total_l1_loss=0

        explainer.train()
        for batch_index, samples in enumerate(loader):
            # prepare data for explainer:
            user_tensors = torch.Tensor(samples[:,:-1]).to(device)
            user_ids = samples[:,-1]
            top1_item = np.array([top1_train[int(x)] for x in user_ids])
            items_vectors = items_array[top1_item]
            items_tensors = torch.Tensor(items_vectors).to(device)
            n = user_tensors.shape[0]

            # zero grad:
            optimizer_comb.zero_grad()
            # forward:
            expl_scores = explainer(user_tensors, items_tensors)

            # caclulate loss
            comb_loss, pos_loss, neg_loss, l1 = loss_func(user_tensors, items_tensors, top1_item, expl_scores)
            train_loss += comb_loss*n
            total_pos_loss += pos_loss*n
            total_neg_loss += neg_loss*n
            total_l1_loss += l1*n

            # back propagation
            comb_loss.backward()
            optimizer_comb.step()

        train_metrics = {"train/train_loss": train_loss,
                         "train/pos_loss": total_pos_loss,
                         "train/neg_loss": total_neg_loss,
                         "train/l1_loss": total_l1_loss,
                         "train/epoch": epoch}

        if epoch % ((epochs*100)/100): 
            print(f"saving the checkpoint in epch {epoch} with filename: 18LXR_{data_name}_{recommender_name}_{trial.number}_{epoch}_{explainer_hidden_size}_{lambda_pos}_{lambda_neg}.pt")
            torch.save(explainer.state_dict(), Path(checkpoints_path, f'18LXR_{data_name}_{recommender_name}_{trial.number}_{epoch}_{explainer_hidden_size}_{lambda_pos}_{lambda_neg}.pt'))

        #Monitoring on POS metric after each epoch
        explainer.eval()
        for j in range(random_sampled_array.shape[0]):
            user_id = random_sampled_array[j][-1]
            user_tensor = torch.Tensor(random_sampled_array[j][:-1]).to(device)
            top1_item_test = top1_test[user_id]
            item_vector = items_array[top1_item_test]
            items_tensor = torch.Tensor(item_vector).to(device)

            res = calculate_pos_neg_k(user_tensor, top1_item_test, items_tensor, num_of_bins, explainer, k)
            POS_at_k_lxr += res[0]
            NEG_at_k_lxr += res[1]

        last_pos_at_k = np.mean(POS_at_k_lxr)/random_sampled_array.shape[0]
        last_neg_at_k = np.mean(NEG_at_k_lxr)/random_sampled_array.shape[0]
        run_pos_at_k.append(last_pos_at_k)
        run_neg_at_k.append(last_neg_at_k)
        metric_for_monitoring.append(last_pos_at_k.item())

        val_metrics = {
            "val/pos_at_k": last_pos_at_k,
            "val/neg_at_k": last_neg_at_k
        }
        
        wandb.log({**train_metrics, **val_metrics})
        print(f'Finished epoch {epoch} with run_pos_at_k {last_pos_at_k} and run_neg_at_k {last_neg_at_k}')
        print(f'Train loss = {train_loss}')
        if epoch>=5: # early stop conditions - if both pos@k and neg@k are getting worse in the past 4 epochs
            if run_pos_at_k[-2]<run_pos_at_k[-1] and run_pos_at_k[-3]<run_pos_at_k[-2] and run_pos_at_k[-4]<run_pos_at_k[-3]:
                if run_neg_at_k[-2]>run_neg_at_k[-1] and run_neg_at_k[-3]>run_neg_at_k[-2] and run_neg_at_k[-4]>run_neg_at_k[-3]:
                    print(f'Early stop at trial with lambda_pos = {lambda_pos}, lambda_neg = {lambda_neg}, alpha_parameter = {alpha}. Best results at epoch {np.argmin(run_pos_at_k)} with value {np.min(run_pos_at_k)}')
                    return np.min(metric_for_monitoring) # return the best pos@k value in this trial

    print(f'Stop at trial with lambda_pos = {lambda_pos}, lambda_neg = {lambda_neg}, alpha_parameter = {alpha}. Best results at epoch {np.argmin(run_pos_at_k)} with value {np.min(run_pos_at_k)}')    
    return np.min(metric_for_monitoring) # return the best pos@k value in this trial

# ### Save logs in text file, optimize using Optuna

# Testing Wandb

import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()

logger = logging.getLogger()

logger.setLevel(logging.INFO)  # Setup the root logger.
logger.addHandler(logging.FileHandler(f"{data_name}_{recommender_name}_explainer_training.log", mode="w"))

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

study = optuna.create_study(direction='minimize')

logger.info("Start optimization.")
study.optimize(lxr_training, n_trials)

with open(f"{data_name}_{recommender_name}_explainer_training.log") as f:
    assert f.readline().startswith("A new study created")
    assert f.readline() == "Start optimization.\n"
    
    
# Print best hyperparameters and corresponding metric value
print("Best hyperparameters: {}".format(study.best_params))
print("Best metric value: {}".format(study.best_value))

wandb.finish()
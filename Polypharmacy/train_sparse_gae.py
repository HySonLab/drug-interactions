import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as pyg_T
import torch_geometric.utils as pyg_utils
from torch_sparse import SparseTensor
from models import SparseGAE
from metrics import *
from data import *
import time
import numpy as np
import scipy.sparse as sp 
import warnings
warnings.filterwarnings("ignore")

#modified
seed = 6
torch_geometric.seed_everything(seed)

print("Load data")
data = load_data(drug_feature = None)
edge_types = data.edge_types
rev_edge_types = []


for (src, relation, dst) in edge_types:
    rev_relation = f"rev_{relation}"
    rev_edge_types.append((dst, rev_relation, src))

transform = pyg_T.Compose([
    pyg_T.RandomLinkSplit(num_val = 0.1, num_test = 0.1, 
                            is_undirected = False, add_negative_train_samples = False,
                            edge_types = edge_types, rev_edge_types = rev_edge_types, 
                            neg_sampling_ratio = 0.0, 
                            disjoint_train_ratio = 0.2),
]
    )

train_data, valid_data, test_data = transform(data)


for node in data.node_types:
    train_data[node].x = train_data[node].x.to_sparse().float()
    valid_data[node].x = valid_data[node].x.to_sparse().float()
    test_data[node].x = test_data[node].x.to_sparse().float()

for edge_type in rev_edge_types:
    del train_data[edge_type]
    del valid_data[edge_type]
    del test_data[edge_type]


for edge_type in edge_types:
    if edge_type[0] == edge_type[2]:
        train_data[edge_type].edge_index = pyg_utils.to_undirected(train_data[edge_type].edge_index)
        valid_data[edge_type].edge_index = pyg_utils.to_undirected(valid_data[edge_type].edge_index)
        test_data[edge_type].edge_index = pyg_utils.to_undirected(test_data[edge_type].edge_index)


device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = pyg_T.ToSparseTensor()(train_data)
valid_data = pyg_T.ToSparseTensor()(valid_data)
test_data  = pyg_T.ToSparseTensor()(test_data)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    if adj.shape[0] == adj.shape[1]:
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    else:
        rowsum = np.array(adj.sum(1))
        colsum = np.array(adj.sum(0))
        rowdegree_mat_inv = sp.diags(np.nan_to_num(np.power(rowsum, -0.5)).flatten())
        coldegree_mat_inv = sp.diags(np.nan_to_num(np.power(colsum, -0.5)).flatten())
        adj_normalized = rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv).tocoo()
    return sparse_to_tuple(adj_normalized)


train_adj_t_dict = {}
valid_adj_t_dict = {}
test_adj_t_dict = {}

for src, relation, dst in train_data.edge_types:
    adj_norm = preprocess_graph(train_data.adj_t_dict[src, relation, dst].t().to_dense())
    train_adj_t_dict[src, relation, dst] = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                                                    torch.FloatTensor(adj_norm[1]), 
                                                                    torch.Size(adj_norm[2])).to(device).t()
    
    adj_norm = preprocess_graph(valid_data.adj_t_dict[src, relation, dst].t().to_dense())
    valid_adj_t_dict[src, relation, dst] = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                                                    torch.FloatTensor(adj_norm[1]), 
                                                                    torch.Size(adj_norm[2])).to(device).t()
    adj_norm = preprocess_graph(test_data.adj_t_dict[src, relation, dst].t().to_dense())
    test_adj_t_dict[src, relation, dst] = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                                                    torch.FloatTensor(adj_norm[1]), 
                                                                    torch.Size(adj_norm[2])).to(device).t()
in_dims = {
    "drug" : 256,
    "gene" : 19081
}
hidden_dims = [64, 32]
num_layer = 2
out_dim = hidden_dims[-1]
decoder_map = {
    "bilinear" : ["interact", "has_target", "get_target"],
    "dedicom" : [relation for (_, relation, _) in edge_types
                    if relation not in ["interact", "has_target", "get_target"]]}

relation_2_decoder = {
    "interact" : "bilinear",
    "has_target" : "bilinear",
    "get_target" : "bilinear",
    }

for (_, relation, _) in edge_types:
    if relation not in ["interact", "has_target", "get_target"]:
        relation_2_decoder[relation] = "dedicom"

model = SparseGAE(in_dims, hidden_dims, out_dim, num_layer, train_data.node_types, train_data.edge_types, 
                      decoder_map, relation_2_decoder, dropout = 0.5, device = device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
loss_fn = nn.BCEWithLogitsLoss(reduction = "sum")

train_data = train_data.to(device)
valid_data = valid_data.to(device)


pos_edge_label_index_dict = train_data.edge_label_index_dict
edge_label_index_dict = {}
edge_label_dict = {}


for epoch in range(600):
    start = time.time()
    model.train()
    pos_edge_label_index_dict = train_data.edge_label_index_dict
    edge_label_index_dict = {}
    edge_label_dict = {}
    optimizer.zero_grad()
    z_dict = model.encode(train_data.x_dict, train_adj_t_dict)
    for edge_type in edge_types:
        src, relation, dst = edge_type
        if relation == "get_target":
            continue
        num_nodes = (train_data.x_dict[src].shape[0], train_data.x_dict[dst].shape[0])
        neg_edge_label_index = pyg_utils.negative_sampling(pos_edge_label_index_dict[edge_type], num_nodes = num_nodes)
        edge_label_index_dict[edge_type] = torch.cat([pos_edge_label_index_dict[edge_type], 
                                                        neg_edge_label_index], dim = -1)

        pos_label = torch.ones(pos_edge_label_index_dict[edge_type].shape[1])
        neg_label = torch.zeros(neg_edge_label_index.shape[1])
        edge_label_dict[relation] = torch.cat([pos_label, neg_label], dim = 0)
    edge_pred = model.decode_all_relation(z_dict, edge_label_index_dict) 

    edge_pred = torch.cat([edge_pred[relation] for relation in edge_pred.keys()], dim = -1)
    edge_label = torch.cat([edge_label_dict[relation] for relation in edge_label_dict.keys()], dim = -1).to(device)
    loss = loss_fn(edge_pred, edge_label)
    loss.backward()
    optimizer.step()
    loss = loss.detach().item()
    
    model.eval()
    with torch.no_grad():
        z_dict = model.encode(valid_data.x_dict, valid_adj_t_dict)
        pos_edge_label_index_dict = valid_data.edge_label_index_dict
        edge_label_index_dict = {}
        edge_label_dict = {}
        for edge_type in edge_types:
            src, relation, dst = edge_type
            if relation == "get_target":
                continue
            num_nodes = (train_data.x_dict[src].shape[0], train_data.x_dict[dst].shape[0])
            neg_edge_label_index = pyg_utils.negative_sampling(pos_edge_label_index_dict[edge_type], num_nodes = num_nodes)
            edge_label_index_dict[edge_type] = torch.cat([pos_edge_label_index_dict[edge_type], 
                                                            neg_edge_label_index], dim = -1)

            pos_label = torch.ones(pos_edge_label_index_dict[edge_type].shape[1])
            neg_label = torch.zeros(neg_edge_label_index.shape[1])
            edge_label_dict[relation] = torch.cat([pos_label, neg_label], dim = 0)
        
        edge_pred = model.decode_all_relation(z_dict, edge_label_index_dict) 
        cat_edge_pred = torch.cat([edge_pred[relation] for relation in edge_pred.keys()], dim = -1)
        cat_edge_label = torch.cat([edge_label_dict[relation] for relation in edge_label_dict.keys()], dim = -1).to(device)
        val_loss = loss_fn(cat_edge_pred, cat_edge_label).item()

        
        for relation in edge_pred.keys():
            edge_pred[relation] = F.sigmoid(edge_pred[relation]).cpu()
        roc_auc, _ = cal_roc_auc_score_per_side_effect(edge_pred, edge_label_dict, edge_types)
        prec, _ = cal_average_precision_score_per_side_effect(edge_pred, edge_label_dict, edge_types)
 
    end = time.time()
    print("| Epoch: {:3d} | Loss: {:5.5f} | Val Loss: {:5.5f} | Val ROC: {:5.5f} | Val Prec: {:5.5f} | Time: {:5.5f}".format(epoch,
                                                                                                                            loss, 
                                                                                                                            val_loss,
                                                                                                                            roc_auc,
                                                                                                                            prec,
                                                                                                                            end - start))


test_data = test_data.to(device)
model.eval()
with torch.no_grad():
    z_dict = model.encode(test_data.x_dict, test_adj_t_dict)
    pos_edge_label_index_dict = test_data.edge_label_index_dict
    edge_label_index_dict = {}
    edge_label_dict = {}
    for edge_type in test_data.edge_label_index_dict.keys():
        src, relation, dst = edge_type
        if relation == "get_target":
                continue
        num_nodes = (test_data.x_dict[src].shape[0], test_data.x_dict[dst].shape[0])
        neg_edge_label_index = pyg_utils.negative_sampling(pos_edge_label_index_dict[edge_type], num_nodes = num_nodes)
        edge_label_index_dict[edge_type] = torch.cat([pos_edge_label_index_dict[edge_type], 
                                                        neg_edge_label_index], dim = -1)

        pos_label = torch.ones(pos_edge_label_index_dict[edge_type].shape[1])
        neg_label = torch.zeros(neg_edge_label_index.shape[1])
        edge_label_dict[relation] = torch.cat([pos_label, neg_label], dim = 0)
    
    edge_pred = model.decode_all_relation(z_dict, edge_label_index_dict)
    for relation in edge_pred.keys():
        edge_pred[relation] = F.sigmoid(edge_pred[relation]).cpu()
    roc_auc, _ = cal_roc_auc_score_per_side_effect(edge_pred, edge_label_dict, edge_types)
    prec, prec_dict = cal_average_precision_score_per_side_effect(edge_pred, edge_label_dict, edge_types)
    apk, apk_dict = cal_apk(edge_pred, edge_label_dict, edge_types, k = 50)
    print("-" * 100)
    print()
    print('Test AUROC: {:5.5f}, Test AUPRC: {:5.5f}, Test AP@50: {:5.5f}'.format(roc_auc, prec, apk))
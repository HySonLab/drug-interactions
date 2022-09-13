import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as pyg_T
import torch_geometric.utils as pyg_utils


from models.hetero_gae import HeteroGAE
from metrics import *
from metrics import *
import time
import numpy as np
from data import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "Polypharmacy Side Effect Prediction")
parser.add_argument("--seed", type = int, default = 1, help = "random seed")
parser.add_argument("--num_epoch", type = int, default = 300, help = "number of epochs")
parser.add_argument("--lr", type = float, default = 1e-3, help = "learning rate")
parser.add_argument("--chkpt_dir", type = str, default = "./checkpoint/", help = "checkpoint directory")
parser.add_argument("--dropout", type = float, default = 0.1, help = "dropout rate")
parser.add_argument("--device", type = str, default = "cuda:0", help = "training device")

args = parser.parse_args()
torch_geometric.seed_everything(args.seed)

print("Load data")
data = load_data()
edge_types = data.edge_types
rev_edge_types = []

for (src, relation, dst) in edge_types:
    rev_relation = f"rev_{relation}"
    rev_edge_types.append((dst, rev_relation, src))

transform = pyg_T.Compose([
    pyg_T.AddSelfLoops(),
    pyg_T.RandomLinkSplit(num_val = 0.1, num_test = 0.1, is_undirected = True,
        edge_types = edge_types, rev_edge_types = rev_edge_types, neg_sampling_ratio = 0.0, disjoint_train_ratio = 0.2)])

train_data, valid_data, test_data = transform(data)

for node in data.node_types:
    train_data[node].x = train_data[node].x.to_sparse().double()
    valid_data[node].x = valid_data[node].x.to_sparse().double()
    test_data[node].x = test_data[node].x.to_sparse().double()

for edge_type in rev_edge_types:
    del train_data[edge_type]
    del valid_data[edge_type]
    del test_data[edge_type]

for edge_type in edge_types:
    if edge_type[0] == edge_type[2]:
        train_data[edge_type].edge_index = pyg_utils.to_undirected(train_data[edge_type].edge_index)
        valid_data[edge_type].edge_index = pyg_utils.to_undirected(valid_data[edge_type].edge_index)
        test_data[edge_type].edge_index = pyg_utils.to_undirected(test_data[edge_type].edge_index)

print("Initialize model...")
hidden_dim = [64, 32]
num_layer = 2

decoder_2_relation = {
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

out_dim = hidden_dim[-1]
net = HeteroGAE(hidden_dim, out_dim, data.node_types, data.edge_types, decoder_2_relation, 
                relation_2_decoder, dropout = args.dropout, device = args.device).to(args.device)
net = net.to(torch.double)
loss_fn = nn.BCEWithLogitsLoss(reduction = "sum")
optimizer = torch.optim.Adam(net.parameters(), lr = args.lr)
num_epoch = args.num_epoch

print("Training device: ", args.device)
train_edge_label_index_dict = train_data.edge_label_index_dict
train_data = train_data.to(args.device)
valid_data = valid_data.to(args.device)
best_val_roc = 0

print("Training...")
for epoch in range(num_epoch):
    start = time.time()
    net.train()
    optimizer.zero_grad()
    z_dict = net.encode(train_data.x_dict, train_data.edge_index_dict)
    pos_edge_label_index_dict = train_data.edge_label_index_dict
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

    edge_pred = net.decode_all_relation(z_dict, edge_label_index_dict) 
    edge_pred = torch.cat([edge_pred[relation] for relation in edge_pred.keys()], dim = -1)
    edge_label = torch.cat([edge_label_dict[relation] for relation in edge_label_dict.keys()], dim = -1).to(args.device)
    loss = loss_fn(edge_pred, edge_label)
    loss.backward()
    optimizer.step()
    loss = loss.detach().item()
    
    net.eval()
    with torch.no_grad():
        z_dict = net.encode(valid_data.x_dict, valid_data.edge_index_dict)
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
        
        edge_pred = net.decode_all_relation(z_dict, edge_label_index_dict)
        for relation in edge_pred.keys():
            edge_pred[relation] = F.sigmoid(edge_pred[relation]).cpu()
        roc_auc, _ = cal_roc_auc_score_per_side_effect(edge_pred, edge_label_dict, edge_types)
    
    end = time.time()
    print(f"| Epoch: {epoch} | Loss: {loss} | Val ROC: {roc_auc} | Best ROC: {best_val_roc} | Time: {end - start}")
    
    if best_val_roc < roc_auc:
        best_val_roc = roc_auc
        torch.save(net.state_dict(), args.chkpt_dir + f"/gae_{args.seed}.pt")
        print("---- Save Model ----")



test_data = test_data.to(args.device)
net.load_state_dict(torch.load(args.chkpt_dir + f"/gae_{args.seed}.pt"))
net.eval()
with torch.no_grad():
    z_dict = net.encode(test_data.x_dict, test_data.edge_index_dict)
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
    
    edge_pred = net.decode_all_relation(z_dict, edge_label_index_dict)
    for relation in edge_pred.keys():
        edge_pred[relation] = F.sigmoid(edge_pred[relation]).cpu()
    roc_auc, _ = cal_roc_auc_score_per_side_effect(edge_pred, edge_label_dict, edge_types)
    prec, prec_dict = cal_average_precision_score_per_side_effect(edge_pred, edge_label_dict, edge_types)
    apk, apk_dict = cal_apk(edge_pred, edge_label_dict, edge_types, k = 50)
    print("-" * 100)
    print()
    print(f'| Test AUROC: {roc_auc} | Test AUPRC: {prec} | Test AP@50: {apk}')
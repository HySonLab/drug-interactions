import os
import pandas as pd 
import json
import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_geometric.data as pyg_data
import torch_geometric.transforms as pyg_T
from metrics import *
import time
from sklearn.model_selection import train_test_split
from tdc.chem_utils import featurize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from read_data import *
from model import *
import warnings
warnings.filterwarnings("ignore")

train_df, valid_df, test_df, drug_2_smiles, drug_2_idx, relation_id = read_files()

train_data = convert_to_hetero_graph(train_df, drug_2_smiles, drug_2_idx, relation_id)

device = "cuda" if torch.cuda.is_available() else "cpu"

decoder_2_relation = {
    "mlp" : [edge_type[1] for edge_type in train_data.edge_types]
}
relation_2_decoder = {
    edge_type[1] : "mlp" for edge_type in train_data.edge_types
}


train_data = train_data.to(device)

train_edge_label_index_dict, train_edge_label_dict = generate_label(train_df, drug_2_idx, relation_id, device)
valid_edge_label_index_dict, valid_edge_label_dict = generate_label(valid_df, drug_2_idx, relation_id, device)
test_edge_label_index_dict, test_edge_label_dict = generate_label(test_df, drug_2_idx, relation_id, device)

print("Num train: ", count_num_labels(train_edge_label_dict))
print("Num valid: ", count_num_labels(valid_edge_label_dict))
print("Num test: ", count_num_labels(test_edge_label_dict))

model = VAE(128, 64, 64, train_data.metadata(), decoder_2_relation, relation_2_decoder,
                device).to(device)

optimizer = torch.optim.AdamW(list(model.parameters()),lr = 1e-3)

kl_lambda = {
    "drug" : 0.001,
    "gene" : 0.001,   
}

for epoch in range(200):
    start = time.time()
    model.train()
    optimizer.zero_grad()
    z_dict = model.encode(train_data.x_dict, train_data.edge_index_dict)
    kl_loss = model.kl_loss_all(reduce = "ratio", lambda_ = kl_lambda)
    recon_loss = model.recon_loss_all_relation(z_dict, train_edge_label_index_dict)
    loss = recon_loss + kl_loss 
    loss.backward()
    optimizer.step()
    loss = loss.detach().item()
  
    model.eval()
    with torch.no_grad():
        z_dict = model.encode(train_data.x_dict, train_data.edge_index_dict)
        pos_edge_label_index_dict = valid_edge_label_index_dict
        edge_label_index_dict = {}
        edge_label_dict = {}
        for id_ in relation_id:
            relation = str(id_)
            src = "drug"
            dst = "drug"
            edge_type = (src, relation, dst)
            num_nodes = train_data.num_nodes
            neg_edge_label_index = pyg_utils.negative_sampling(pos_edge_label_index_dict[relation], num_nodes = num_nodes)
            edge_label_index_dict[relation] = torch.cat([pos_edge_label_index_dict[relation], 
                                                            neg_edge_label_index], dim = -1)

            pos_label = torch.ones(pos_edge_label_index_dict[relation].shape[1])
            neg_label = torch.zeros(neg_edge_label_index.shape[1])
            edge_label_dict[relation] = torch.cat([pos_label, neg_label], dim = 0)
        
        edge_pred = model.decode_all_relation(z_dict, edge_label_index_dict) 
  
        for relation in edge_pred.keys():
            edge_pred[relation] = F.sigmoid(edge_pred[relation]).cpu()
        roc_auc = cal_roc_auc_score(edge_pred, edge_label_dict, relation_id)
        acc = cal_acc_score(edge_pred, edge_label_dict, relation_id)

    end = time.time()
    print("| Epoch: {:3d} | Loss: {:5.5f} | Val ROC: {:5.5f} | Val Acc: {:5.5f} | Time: {:5.5f}".format(epoch,
                                                                                                        loss, 
                                                                                                        roc_auc,
                                                                                                        acc,
                                                                                                        end - start))
model.eval()
with torch.no_grad():
    z_dict = model.encode(train_data.x_dict, train_data.edge_index_dict)

    pos_edge_label_index_dict = test_edge_label_index_dict
    edge_label_index_dict = {}
    edge_label_dict = {}
    for id_ in relation_id:
        relation = str(id_)
        src = "drug"
        dst = "drug"
        edge_type = (src, relation, dst)
        num_nodes = train_data.num_nodes
        neg_edge_label_index = pyg_utils.negative_sampling(pos_edge_label_index_dict[relation], num_nodes = num_nodes)
        edge_label_index_dict[relation] = torch.cat([pos_edge_label_index_dict[relation], 
                                                        neg_edge_label_index], dim = -1)

        pos_label = torch.ones(pos_edge_label_index_dict[relation].shape[1])
        neg_label = torch.zeros(neg_edge_label_index.shape[1])
        edge_label_dict[relation] = torch.cat([pos_label, neg_label], dim = 0)
    
    edge_pred = model.decode_all_relation(z_dict, edge_label_index_dict) 

    for relation in edge_pred.keys():
        edge_pred[relation] = F.sigmoid(edge_pred[relation]).cpu()
    roc_auc = cal_roc_auc_score(edge_pred, edge_label_dict, relation_id)
    acc = cal_acc_score(edge_pred, edge_label_dict, relation_id)
    auprc = cal_average_precision_score(edge_pred, edge_label_dict)

print('Test ROC: {:5.5f} | Test Acc: {:5.5f} | Test AUPRC: {:5.5f}'.format(roc_auc, acc, auprc))

z = z_dict["drug"].cpu().numpy()
np.save("drug_embedding.npy", z)
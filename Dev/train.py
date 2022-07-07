import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as pyg_T
import torch_geometric.utils as pyg_utils
from torch.utils.data import DataLoader

from test_util import *
from model import *
from edge_loader import *

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def get_accuracy_score(preds, labels, edge_types):
    all_labels = torch.cat([labels[relation] for (_, relation, _) in edge_types], dim = -1)
    all_preds = torch.cat([preds[relation] for (_, relation, _) in edge_types], dim = -1)
    all_preds = torch.where(0.5 <= all_preds, 1, 0)

    return accuracy_score(all_labels, all_preds)

def get_roc_score(preds, labels, edge_types):
    all_labels = torch.cat([labels[relation] for (_, relation, _) in edge_types], dim = -1)
    all_preds = torch.cat([preds[relation] for (_, relation, _) in edge_types], dim = -1)

    return roc_auc_score(all_labels, all_preds)
    

print("Load Data.....")
data = generate_toy_example(n_genes = 500, n_drugs = 400) 
print(data)
print(data.is_undirected())
#
print("Split data into train, valid, and test...")
# edge splitting : train, validation, and test
edge_types = data.edge_types 
rev_edge_types = []

for (src, relation, dst) in edge_types:
    rev_relation = f"rev_{relation}"
    rev_edge_types.append((dst, rev_relation, src))

transform = pyg_T.Compose([
    pyg_T.AddSelfLoops(),
    pyg_T.RandomLinkSplit(num_val = 0.1, num_test = 0.2, is_undirected = True, 
        edge_types = edge_types, rev_edge_types = rev_edge_types, neg_sampling_ratio = 0.0)
])

train_data, valid_data, test_data = transform(data)

train_edge_label_index_dict = train_data.edge_label_index_dict

batch_size = 128
minibatch = EdgeBatchIterator(train_edge_label_index_dict, edge_types, batch_size = batch_size)


print("Initialize model...")
# model definition
hidden_dim = [64, 32]
num_layer = 2 
conv_dicts = generate_hetero_conv_dict(hidden_dim, edge_types, num_layer)
decoder_map = {
    "bilinear" : ["interact", "has_target", "get_target"], # drug-target, protein-protein
    "dedicom" : ["0", "1", "2"] # drug-drug
}
relation_2_decoder = {
    "interact" : "bilinear",
    "has_target" : "bilinear",
    "get_target" : "bilinear",
    "0" : "dedicom",
    "1" : "dedicom",
    "2" : "dedicom"
}

out_dim = hidden_dim[-1]
device = "cuda" if torch.cuda.is_available() else "cpu"
net = Decagon(conv_dicts, out_dim, decoder_map, relation_2_decoder, use_sigmoid = False, device = device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)

#a sample of training loop
print("Training.....")
for epoch in range(30):

    net.train()
    total_loss = []
    minibatch.shuffle()
    while not minibatch.end():
        pos_edge_index, edge_type = minibatch.next_minibatch_edge_type()
        src, relation, dst = edge_type
        num_src_node, num_dst_node = data.x_dict[src].shape[1], data.x_dict[dst].shape[1]
    
        optimizer.zero_grad()
        z_dict = net.encode(data.x_dict, train_data.edge_index_dict)
        neg_edge_index = pyg_utils.negative_sampling(pos_edge_index, num_nodes = (num_src_node, num_dst_node))
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim = -1)
        output = net.decode(z_dict, edge_index, edge_type)

        pos_label = torch.ones(pos_edge_index.shape[1])
        neg_label = torch.zeros(neg_edge_index.shape[1])
        label = torch.cat([pos_label, neg_label], dim = 0)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.detach().item())

    # Validation
    net.eval()
    val_outputs = {}
    val_labels = {}
    with torch.no_grad():
        for edge_type in edge_types:
            src, relation, dst = edge_type
            num_src_node, num_dst_node = data.x_dict[src].shape[1], data.x_dict[dst].shape[1]
            z_dict = net.encode(data.x_dict, train_data.edge_index_dict)
            val_pos_edge_index = valid_data.edge_label_index_dict[edge_type]
            val_neg_edge_index = pyg_utils.negative_sampling(val_pos_edge_index, num_nodes = (num_src_node, num_dst_node))
            val_edge_index = torch.cat([val_pos_edge_index, val_neg_edge_index], dim = -1)
            val_outputs[relation] = F.sigmoid(net.decode(z_dict, val_edge_index, edge_type))
            #sigmoid

            val_pos_label = torch.ones(val_pos_edge_index.shape[1])
            val_neg_label = torch.zeros(val_neg_edge_index.shape[1])
            val_labels[relation] = torch.cat([val_pos_label, val_neg_label], dim = 0)

    acc = get_accuracy_score(val_outputs, val_labels, edge_types)
    roc_auc = get_roc_score(val_outputs, val_labels, edge_types) 
    print("Epoch: {}, Train Loss: {}, Val ROC: {}, Val Accuracy: {}".format(epoch, np.array(total_loss).mean(), roc_auc, acc))

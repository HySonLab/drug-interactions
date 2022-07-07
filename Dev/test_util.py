import torch
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_util

from torch_geometric.data import HeteroData
from operator import itemgetter
from itertools import combinations

import numpy as np
import networkx as nx
import scipy.sparse as sp

def generate_toy_example(n_genes = 500, n_drugs = 400, n_drugdrug_rel_types = 3):
    gene_net = nx.path_graph(n_genes)
    gene_adj = nx.adjacency_matrix(gene_net)
    gene_drug_adj = sp.csr_matrix((np.random.randn(n_genes, n_drugs) > 15 / 10).astype(int))
    drug_gene_adj = gene_drug_adj.transpose(copy=True)

    drug_drug_adj_list = []
    drug_drug_edge_index_list = []
    tmp = np.dot(drug_gene_adj, gene_drug_adj)
    for i in range(n_drugdrug_rel_types):
        mat = np.zeros((n_drugs, n_drugs))
        for d1, d2 in combinations(list(range(n_drugs)), 2):
            if tmp[d1, d2] == i + 4:
                mat[d1, d2] = mat[d2, d1] = 1.
        drug_drug_adj_list.append(sp.csr_matrix(mat))
        edge_index, _ = pyg_util.from_scipy_sparse_matrix(sp.csr_matrix(mat))
        drug_drug_edge_index_list.append(edge_index)

    drug_gene_edge_index, _ = pyg_util.from_scipy_sparse_matrix(drug_gene_adj)
    gene_drug_edge_index, _ = pyg_util.from_scipy_sparse_matrix(gene_drug_adj)
    gene_edge_index, _ = pyg_util.from_scipy_sparse_matrix(gene_adj)

    data = HeteroData()

    data["gene"].x = torch.eye(n_genes)
    data["drug"].x = torch.eye(n_drugs)
    data["gene", "interact", "gene"].edge_index = gene_edge_index
    data["drug", "has_target", "gene"].edge_index = drug_gene_edge_index
    data["gene", "get_target", "drug"].edge_index = gene_drug_edge_index
    for i in range(n_drugdrug_rel_types):
        data["drug", str(i), "drug"].edge_index = drug_drug_edge_index_list[i]

    return data

def generate_hetero_conv_dict(hidden_dims, edge_types, num_layer):
    conv_dicts = []

    for i in range(num_layer):
        D = {}
        for edge_type in edge_types:
            src, relation, dst = edge_type
            if src == dst:
                D[edge_type] = pyg_nn.GCNConv(-1,hidden_dims[i])
            else:
                D[edge_type] = pyg_nn.GeneralConv((-1,-1), hidden_dims[i], l2_normalize = True)
        conv_dicts.append(D)

    return conv_dicts
            
            

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from models.decoder_module import *

def reset_params(self):
    self.lin_msg.weight_initializer = "glorot"
    self.lin_msg.reset_parameters()

    if hasattr(self.lin_self, "reset_parameters"):
        self.lin_self.weight_initializer = "glorot"
        self.lin_self.reset_parameters()

    if not self.directed_msg:
        self.lin_msg_i.weight_initializer = "glorot"
        self.lin_msg_i.reset_parameters()

    if self.in_edge_channels is not None:
        self.lin_edge.weight_initializer = "glorot"
        self.lin_edge_weight.reset_parameters()

GeneralConv.reset_parameters = reset_params

def generate_hetero_conv_dict(hidden_dims, edge_types, num_layer):
    conv_dicts = []

    for i in range(num_layer):
        D = {}
        for edge_type in edge_types:
            src, relation, dst = edge_type
            D[edge_type] = GeneralConv((-1,-1), hidden_dims[i], aggr = "sum", skip_linear = True, l2_normalize = True)
        conv_dicts.append(D)
    return conv_dicts


class HeteroGAE(nn.Module):
    def __init__(self, hidden_dims, out_dim, node_types, edge_types,
            decoder_2_relation, relation_2_decoder, dropout = 0.5, device = "cpu"):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.edge_types = edge_types
        self.node_types = node_types
        self.decoder_2_relation = decoder_2_relation
        self.relation_2_decoder = relation_2_decoder

        self.encoder = nn.ModuleList()
        conv_dicts = generate_hetero_conv_dict(self.hidden_dims, self.edge_types,len(self.hidden_dims))
        for i in range(len(conv_dicts)):
            conv = pyg_nn.HeteroConv(conv_dicts[i], aggr = "sum")
            self.encoder.append(conv)

        self.decoder = nn.ModuleDict()

        for decoder_type in self.decoder_2_relation.keys():
            if decoder_type == "bilinear":
                self.decoder[decoder_type] = BilinearDecoder(out_dim, self.decoder_2_relation["bilinear"], device)
            elif decoder_type == "dedicom":
                self.decoder[decoder_type] = DEDICOM(out_dim, self.decoder_2_relation["dedicom"], device)
            else:
                raise NotImplemented
        self.dropout = dropout

    def forward(self):
        pass

    def encode(self, x_dict, edge_index_dict):
        z_dict = x_dict
        for idx, conv in enumerate(self.encoder):
            z_dict = conv(z_dict, edge_index_dict)
            if idx < len(self.encoder) - 1:
                z_dict = {key : x.relu() for key, x in z_dict.items()}
            z_dict = {key : F.dropout(x, p = self.dropout, training = self.training)
                        for key, x in z_dict.items()}
        return z_dict

    def decode_all_relation(self, z_dict, edge_index_dict, sigmoid = False):
        output = {}
        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict.keys():
                continue
            src, relation, dst = edge_type
            decoder_type = self.relation_2_decoder[relation]
            z = (z_dict[src], z_dict[dst])
            output[relation] = self.decoder[decoder_type](z, edge_index_dict[edge_type], relation)
            if sigmoid:
                output[relation] = F.sigmoid(output[relation])
        return output

    def decode(self, z_dict, edge_index, edge_type, sigmoid = False):
        src, relation, dst = edge_type
        decoder_type = self.relation_3_decoder[relation]
        z = (z_dict[src], z_dict[dst])
        output = self.decoder[decoder_type](z, edge_index, relation)
        if sigmoid:
            output = F.sigmoid(output)
        return output
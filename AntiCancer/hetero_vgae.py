import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv, GraphConv
from torch_geometric.nn.inits import glorot
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from decoder_module import *

# override reset_parameters() function to be compatible with glorot initializer.
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
        self.lin_edge.reset_parameters()

    if self.attention and self.attention_type == "additive":
        glorot(self.att_msg)

GeneralConv.reset_parameters = reset_params
def generate_hetero_conv_dict(hidden_dims, edge_types, num_layer):
    conv_dicts = []

    for i in range(num_layer):
        D = {}
        for edge_type in edge_types:
            src, relation, dst = edge_type
            #D[edge_type] = GeneralConv((-1,-1), hidden_dims[i], l2_normalize = True, in_edge_channels=1)
            D[edge_type] = GraphConv((-1, -1), hidden_dims[i], aggr = "mean")
        conv_dicts.append(D)
    return conv_dicts


class HeteroVGAE(nn.Module):
    def __init__(self, input_dims, hidden_dims, out_dim, node_types, edge_types,
                decoder_2_relation, relation_2_decoder, latent_encoder_type = "linear", dropout = 0.1, device = "cpu"):
        super().__init__()

        assert isinstance(hidden_dims, list)

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.edge_types = edge_types
        self.node_types = node_types
        self.decoder_2_relation = decoder_2_relation
        self.relation_2_decoder = relation_2_decoder

        self.fc1 = nn.ModuleDict()
        self.fc2 = nn.ModuleDict()

        self.encoder = nn.ModuleList()
        conv_dicts = generate_hetero_conv_dict(self.hidden_dims, self.edge_types,len(self.hidden_dims))
        for i in range(len(conv_dicts)):
            conv = pyg_nn.HeteroConv(conv_dicts[i], aggr = "sum")
            self.encoder.append(conv)

        if latent_encoder_type == "gconv":
            mu_latent_encoder_dict = generate_hetero_conv_dict([self.out_dim], self.edge_types, 1)
            self.mu_latent_encoder = pyg_nn.HeteroConv(mu_latent_encoder_dict[0], aggr = "sum")
            logstd_latent_encoder_dict = generate_hetero_conv_dict([self.out_dim], self.edge_types, 1)
            self.logstd_latent_encoder = pyg_nn.HeteroConv(logstd_latent_encoder_dict[0], aggr = "sum")
        elif latent_encoder_type == "linear":
            self.mu_latent_encoder = nn.ModuleDict()
            self.logstd_latent_encoder = nn.ModuleDict()
            for node_type in self.node_types:
                self.mu_latent_encoder[node_type] = nn.Sequential(
                                                        nn.Linear(self.hidden_dims[-1], self.out_dim // 2), 
                                                        nn.ReLU(),
                                                        nn.Linear(self.out_dim // 2, self.out_dim)).to(device)
                self.logstd_latent_encoder[node_type] = nn.Sequential(
                                                        nn.Linear(self.hidden_dims[-1], self.out_dim // 2), 
                                                        nn.ReLU(),
                                                        nn.Linear(self.out_dim // 2, self.out_dim)).to(device)

        self.latent_encoder_type = latent_encoder_type       

        self.decoder = nn.ModuleDict()

        for decoder_type in self.decoder_2_relation.keys():
            if decoder_type == "bilinear":
                self.decoder[decoder_type] = BilinearDecoder(self.out_dim, self.decoder_2_relation["bilinear"], device)
            elif decoder_type == "dedicom":
                self.decoder[decoder_type] = DEDICOM(self.out_dim, self.decoder_2_relation["dedicom"], device)
            elif decoder_type == "inner":
                self.decoder[decoder_type] = InnerProductDecoder()
            elif decoder_type == "linear":
                self.decoder[decoder_type] = EdgeDecoder(self.out_dim)
            else:
                raise NotImplemented

        self.dropout = dropout
        self.device = device

        self.__mu__ = {}
        self.__logstd__ = {}

    def reparameterize(self, mu, logstd):
        z_dict = {}
        for node_type in self.node_types:
            z_dict[node_type] = mu[node_type] + torch.randn_like(logstd[node_type]) * torch.exp(logstd[node_type])
        return z_dict

    def encode(self, x_dict, edge_index_dict, edge_attr_dict):
        z_dict = x_dict
        for idx, conv in enumerate(self.encoder):
            z_dict = conv(z_dict, edge_index_dict, edge_weight_dict =edge_attr_dict)
            z_dict = {key : F.normalize(z, p = 2) for key, z in z_dict.items()}
            z_dict = {key : z.relu() for key, z in z_dict.items()}
            z_dict = {key : F.dropout(z, p = self.dropout, training = self.training)
                        for key, z in z_dict.items()}

        if self.latent_encoder_type == "gconv":
            self.__mu__ = self.mu_latent_encoder(z_dict, edge_index_dict)
            self.__logstd__ = self.logstd_latent_encoder(z_dict, edge_index_dict)
        elif self.latent_encoder_type == "linear":
            for node_type in self.node_types:
                self.__mu__[node_type] = self.mu_latent_encoder[node_type](z_dict[node_type])
                self.__logstd__[node_type] = self.logstd_latent_encoder[node_type](z_dict[node_type])

        for node_type in self.node_types:
            self.__logstd__[node_type] = self.__logstd__[node_type].clamp(max = 10)

        return self.reparameterize(self.__mu__, self.__logstd__)

    def decode(self, z_dict, edge_index, edge_type, sigmoid = False):
        src, relation, dst = edge_type
        decoder_type = self.relation_2_decoder[relation]
        z = (z_dict[src], z_dict[dst])
        output = self.decoder[decoder_type](z, edge_index, relation)
        if sigmoid:
            output = F.sigmoid(output)
        return output

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

    def kl_loss(self, node_type, mu = None, logstd = None):
        mu = self.__mu__[node_type] if mu is None else mu
        logstd = self.__logstd__[node_type] if logstd is None else logstd.clamp(max=10)
        return -0.5 * torch.mean(
                torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim = 1))

    def kl_loss_all(self, reduce = "sum", ratio = None):
        assert len(self.__mu__) > 0 and len(self.__logstd__) > 0
        loss = {}
        for node_type in self.node_types:
            loss[node_type] = self.kl_loss(node_type, self.__mu__[node_type], self.__logstd__[node_type])
        if reduce == "sum":
            loss = sum(loss.values())
        elif reduce == "ratio":
            assert isinstance(ratio, dict)
            loss = sum([loss[node_type] * ratio[node_type] for node_type in ratio.keys()])
        else:
            raise NotImplemented
        return loss

    def recon_loss(self, z_dict, edge_type, pos_edge_index, neg_edge_index = None):
        poss_loss = -torch.log(
                self.decode(z_dict, pos_edge_index, edge_type, sigmoid= True) + 1e-15).mean()

        if neg_edge_index is None:
            src, relation, dst = edge_type
            num_src_node, num_dst_node = z_dict[src].shape[0], z_dict[dst].shape[0]
            neg_edge_index = pyg_utils.negative_sampling(pos_edge_index, num_nodes = (num_src_node, num_dst_node))

        neg_loss = -torch.log(1 -
                self.decode(z_dict, neg_edge_index, edge_type, sigmoid=True) + 1e-15).mean()

        return poss_loss + neg_loss

    def recon_loss_all_relation(self, z_dict, edge_index_dict, reduce = "sum"):
        loss = {}
        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict.keys():
                continue
            loss[edge_type] = self.recon_loss(z_dict, edge_type, edge_index_dict[edge_type])

        if reduce == "sum":
            loss = sum(loss.values())
        elif reduce == "mean":
            loss = sum(loss.values()) / len(self.edge_types)
        return loss
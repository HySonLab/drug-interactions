import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_geometric.data as pyg_data
import torch_geometric.transforms as pyg_T
from decoder_module import *


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = pyg_nn.SAGEConv((-1, -1), hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)
        self.conv2 = pyg_nn.SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.norm(self.conv1(x, edge_index)).relu()
        x = self.conv2(x, edge_index)
        return x


class VAE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, latent_dim, metadata, 
                        decoder_2_relation, relation_2_decoder, device):
        super().__init__()

        self.node_type = metadata[0]
        self.hetero_model = GNN(hidden_channels=hidden_channels, out_channels=out_channels)
        self.hetero_model = pyg_nn.to_hetero(self.hetero_model, metadata, aggr='mean')

        self.mu_latent_encoder = nn.ModuleDict()
        self.logstd_latent_encoder = nn.ModuleDict()
        for node_type in self.node_type:
            self.mu_latent_encoder[node_type] = nn.Sequential(
                                                        nn.Linear(out_channels, latent_dim * 2), 
                                                        nn.LayerNorm(latent_dim * 2),
                                                        nn.Tanh(),
                                                        nn.Linear(latent_dim * 2, latent_dim))

            self.logstd_latent_encoder[node_type] = nn.Sequential(
                                                        nn.Linear(out_channels, latent_dim * 2), 
                                                        nn.LayerNorm(latent_dim * 2),
                                                        nn.Tanh(),
                                                        nn.Linear(latent_dim * 2, latent_dim))

            

        self.__mu__ = {}
        self.__logstd__ = {}


        self.decoder = nn.ModuleDict()
        self.decoder_2_relation = decoder_2_relation
        self.relation_2_decoder = relation_2_decoder

        for decoder_type in self.decoder_2_relation.keys():
            if decoder_type == "bilinear":
                self.decoder[decoder_type] = BilinearDecoder(latent_dim , self.decoder_2_relation["bilinear"], device)
            elif decoder_type == "dedicom":
                self.decoder[decoder_type] = DEDICOM(latent_dim , self.decoder_2_relation["dedicom"], device)
            elif decoder_type == "inner":
                self.decoder[decoder_type] = InnerProductDecoder()
            elif decoder_type == "mlp":
                self.decoder[decoder_type] = MLP_Decoder(latent_dim, self.decoder_2_relation["mlp"])
            else:
                raise NotImplemented

    def reparameterize(self, mu, logstd):
        z_dict = {}
        for node_type in self.node_type:
            z_dict[node_type] = mu[node_type] + torch.randn_like(logstd[node_type]) * torch.exp(logstd[node_type])
        return z_dict

    def encode(self, x_dict, edge_index_dict):
        z_dict = self.hetero_model(x_dict, edge_index_dict)
        
        for node_type in self.node_type:
            self.__mu__[node_type] = self.mu_latent_encoder[node_type](z_dict[node_type])
            self.__logstd__[node_type] = self.logstd_latent_encoder[node_type](z_dict[node_type])


        for node_type in self.node_type:
            self.__logstd__[node_type] = self.__logstd__[node_type].clamp(max = 10)

        return self.reparameterize(self.__mu__, self.__logstd__)

    def decode(self, z_dict, edge_index, edge_type = None, sigmoid = False):
        src, relation, dst = edge_type
        decoder_type = self.relation_2_decoder[relation]
        z = (z_dict[src], z_dict[dst])
        output = self.decoder[decoder_type](z, edge_index, relation)
        if sigmoid:
            output = F.sigmoid(output)
        return output

    def decode_all_relation(self, z_dict, edge_index_dict, sigmoid = False):
        output = {}
        for edge_type in edge_index_dict.keys():
            src, relation, dst = "drug", edge_type, "drug"
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

    def kl_loss_all(self, reduce = "sum", lambda_ = None):
        assert len(self.__mu__) > 0 and len(self.__logstd__) > 0
        loss = {}
        for node_type in self.node_type:
            loss[node_type] = self.kl_loss(node_type, self.__mu__[node_type], self.__logstd__[node_type])
        if reduce == "sum":
            loss = sum(loss.values())
        elif reduce == "ratio":
            assert lambda_ is not None
            loss = sum([loss[node_type] * lambda_[node_type] for node_type in self.node_type])
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
        for relation in edge_index_dict.keys():
            edge_type = "drug", relation, "drug"
            loss[edge_type] = self.recon_loss(z_dict, edge_type, edge_index_dict[relation])

        if reduce == "sum":
            loss = sum(loss.values())
        elif reduce == "mean":
            loss = sum(loss.values()) / len(edge_index_dict.keys())
        return loss
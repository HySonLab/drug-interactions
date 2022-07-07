import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class BilinearDecoder(nn.Module):
    def __init__(self, dim, relations, device):
        super().__init__()
        
        self.M = nn.ParameterDict()

        for relation in relations:
            self.M[relation] = nn.Parameter(torch.empty((dim, dim))).to(device)

        self.init_weights()
        
    def forward(self, z, edge_index, relation):
        if isinstance(z, tuple):
            src = z[0][edge_index[0]]
            dst = z[1][edge_index[1]]
        else:
            src = z[edge_index[0]]
            dst = z[edge_index[1]]
        out = torch.matmul(src, self.M[relation])
        return (out * dst).sum(dim = 1)

    def init_weights(self):
        for relation in self.M.keys():
            self.M[relation] = nn.init.xavier_uniform_(self.M[relation])


class DEDICOM(nn.Module):
    def __init__(self, dim, relations, device):
        super().__init__()

        self.R = nn.Parameter(torch.empty((dim, dim))).to(device)
        self.D = nn.ParameterDict()

        for relation in relations:
            self.D[relation] = nn.Parameter(torch.empty(dim, 1)).to(device)

        self.init_weights()

    def forward(self, z, edge_index, relation):
        if isinstance(z, tuple):
            src = z[0][edge_index[0]]
            dst = z[1][edge_index[0]]
        else:
            src = z[edge_index[0]]
            dst = z[edge_index[1]]
        
        out = torch.matmul(src, torch.diag(self.D[relation].squeeze()))
        out = torch.matmul(out, self.R)
        out = torch.matmul(out, torch.diag(self.D[relation].squeeze()))
        out = (out * dst).sum(dim = 1)
        return out

    def init_weights(self):
        self.R = nn.init.xavier_uniform_(self.R)

        for relation in self.D.keys():
            self.D[relation] = nn.init.xavier_uniform_(self.D[relation])


class Decagon(nn.Module):
    def __init__(self, conv_dicts, out_dim, decoder_2_relation, relation_2_decoder, 
                use_sigmoid = False, device = "cpu"):
        super().__init__()

        assert len(conv_dicts) > 0

        self.decoder_2_relation = decoder_2_relation
        self.relation_2_decoder = relation_2_decoder
        self.edge_types = conv_dicts[0].keys()
        self.encoder = nn.ModuleList()
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

        self.use_sigmoid = use_sigmoid

    def forward(self, x_dict, edge_index_dict):
        for conv in self.encoder:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key : x.relu() for key, x in x_dict.items()}

        output = {}
            
        for decoder_type in self.decoder_2_relation.keys():
            for edge_type in self.edge_types:
                src, relation, dst = edge_type
                if relation not in self.relation_2_decoder.keys():
                    continue
                x = (x_dict[src], x_dict[dst])
                output[relation] = self.decoder[decoder_type](x, edge_index_dict[src, relation, dst], relation)
                if self.use_sigmoid:
                    output[relation] = F.sigmoid(output[relation])
        return output

    def encode(self, x_dict, edge_index_dict):
        z_dict = None
        for conv in self.encoder:
            z_dict = conv(x_dict, edge_index_dict)
            z_dict = {key : x.relu() for key, x in z_dict.items()}
        return z_dict

    def decode_all_relation(self, z_dict, edge_index_dict):
        output = {}
        for decoder_type in self.decoder_2_relation.keys():
            for edge_type in self.edge_types:
                src, relation, dst = edge_type
                if relation not in self.relation_2_decoder.keys():
                    continue
                z = (z_dict[src], z_dict[dst])
                output[relation] = self.decoder[decoder_type](z, edge_index_dict[src, relation, dst], relation)
                if self.use_sigmoid:
                    output[relation] = F.sigmoid(output[relation])
        return output

    def decode(self, z_dict, edge_index, edge_type):
        src, relation, dst = edge_type
        decoder_type = self.relation_2_decoder[relation]
        z = (z_dict[src], z_dict[dst])
        output = self.decoder[decoder_type](z, edge_index, relation)
        if self.use_sigmoid:
            output = F.sigmoid(output)
        return output 

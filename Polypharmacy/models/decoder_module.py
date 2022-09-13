import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class InnerProductDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, edge_index, relation = None):
        if isinstance(z, tuple):
            src = z[0][edge_index[0]]
            dst = z[1][edge_index[1]]
        else:
            src = z[edge_index[0]]
            dst = z[edge_index[1]]
        out = (src * dst).sum(dim = 1)
        return out

class BilinearDecoder(nn.Module):
    def __init__(self, dim, relations, device):
        super().__init__()
        
        self.M = nn.ParameterDict()

        for relation in relations:
            self.M[relation] = nn.Parameter(torch.empty((dim, dim)).to(device))

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

        self.R = nn.Parameter(torch.empty((dim, dim)).to(device))
        self.D = nn.ParameterDict()

        for relation in relations:
            self.D[relation] = nn.Parameter(torch.empty(dim, 1).to(device))

        self.init_weights()

    def forward(self, z, edge_index, relation):
        if isinstance(z, tuple):
            src = z[0][edge_index[0]]
            dst = z[1][edge_index[1]]
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

import torch
from torch.utils.data import Dataset
import numpy as np


class EdgeBatchIterator:
    def __init__(self, edge_index_dict, edge_types, batch_size = 128):
        self.edge_index_dict = edge_index_dict
        self.edge_types = edge_types
        self.batch_size = batch_size
        self.num_edge_types = len(self.edge_types)

        self.iter_ = 0
        self.free_batch_edge_types = list(range(self.num_edge_types))
        self.batch_num = [0] * self.num_edge_types
        self.current_edge_type_idx = 0
        self.edge_type2idx = {}
        self.idx2edge_type = {}
        idx = 0

        for edge_type in self.edge_types:
            self.edge_type2idx[edge_type] = idx
            self.idx2edge_type[idx] = edge_type
            idx += 1

    def end(self):
        finished = len(self.free_batch_edge_types) == 0
        return finished

    def next_minibatch_edge_type(self):
        while True:
            if self.iter_ % 4 == 0:
                #self.current_edge_type_idx = self.edge_type2idx[("gene", "interact", "gene")]
                self.current_edge_type_idx = self.edge_type2idx[self.edge_types[self.iter_ % 4]]
            elif self.iter_ % 4 == 1:
                #self.current_edge_type_idx = self.edge_type2idx[("drug", "has_target", "gene")]
                self.current_edge_type_idx = self.edge_type2idx[self.edge_types[self.iter_ % 4]]
            elif self.iter_ % 4 == 2:
                #self.current_edge_type_idx = self.edge_type2idx[("gene", "get_target", "drug")]
                self.current_edge_type_idx = self.edge_type2idx[self.edge_types[self.iter_ % 4]]
            else:
                if len(self.free_batch_edge_types) > 0:
                    self.current_edge_type_idx = np.random.choice(self.free_batch_edge_types)
                else:
                    #self.current_edge_type_idx = self.edge_type2idx[("gene", "interact", "gene")]
                    self.current_edge_type_idx = self.edge_type2idx[self.edge_types[0]]
                    self.iter_ = 0
            
            edge_type = self.idx2edge_type[self.current_edge_type_idx]
            if self.batch_num[self.current_edge_type_idx] * self.batch_size  <= self.edge_index_dict[edge_type].shape[1] - self.batch_size + 1:
                    break
            else:
                if self.iter_ % 4 in [0,1,2]:
                    self.batch_num[self.current_edge_type_idx] = 0
                else:
                    self.free_batch_edge_types.remove(self.current_edge_type_idx)

        self.iter_ += 1
        start = self.batch_num[self.current_edge_type_idx] * self.batch_size
        self.batch_num[self.current_edge_type_idx] += 1
        batch_edge_index = self.edge_index_dict[edge_type][:, start : start + self.batch_size]
        return batch_edge_index, edge_type

    def shuffle(self):
        for edge_type in self.edge_types:
            random_order = torch.randperm(self.edge_index_dict[edge_type].shape[1])
            self.edge_index_dict[edge_type] = self.edge_index_dict[edge_type][:, random_order]
            self.batch_num[self.edge_type2idx[edge_type]] = 0

        self.current_edge_type_idx = 0
        self.free_batch_edge_types = list(range(self.num_edge_types))
#        self.free_batch_edge_types.remove(self.edge_type2idx[("gene", "interact", "gene")])
#        self.free_batch_edge_types.remove(self.edge_type2idx[("drug", "has_target", "gene")])
#        self.free_batch_edge_types.remove(self.edge_type2idx[("gene", "get_target", "drug")])
        
        self.free_batch_edge_types.remove(self.edge_type2idx[self.edge_types[0]])
        self.free_batch_edge_types.remove(self.edge_type2idx[self.edge_types[1]])
        self.free_batch_edge_types.remove(self.edge_type2idx[self.edge_types[2]])

        self.iter_ = 0

import pandas as pd 
import torch
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils
import numpy as np
from tdc.chem_utils import featurize

def read_files():
    train_df = pd.read_csv("data/ddi_training.csv")
    valid_df = pd.read_csv("data/ddi_validation.csv")
    test_df = pd.read_csv("data/ddi_test.csv")

    drug_smiles = pd.read_csv("data/drug_smiles.csv")
    relation_info = pd.read_csv("data/Interaction_information.csv")

    drugs = drug_smiles["drug_id"].tolist()
    smiles = drug_smiles["smiles"].tolist()

    drug_2_smiles = {
        drug : smi for (drug, smi) in zip(drugs, smiles)
    }

    drugs.sort()

    drug_2_idx = {
        drug : idx for (drug, idx) in zip(drugs, range(len(drugs)))
    }

    relation_id = relation_info["DDI type"].apply(lambda x: int(x.split()[-1]) - 1).tolist()

    return train_df, valid_df, test_df, drug_2_smiles, drug_2_idx, relation_id

def generate_morgan_fingerprint(stitch_2_smile, stitch_2_idx):
    num_drugs = len(stitch_2_idx)
    x = np.identity(num_drugs)
    features = [0 for i in range(num_drugs)]
    for stitch, idx in stitch_2_idx.items():
        features[idx] = featurize.molconvert.smiles2ECFP4(stitch_2_smile[stitch]) 
    augment = np.array(features)
    return augment

def convert_to_hetero_graph(df, drug_2_smiles, drug_2_idx, relation_id):
    data = pyg_data.HeteroData()
    
    drug_name = list(drug_2_idx.keys())


    morgan = generate_morgan_fingerprint(drug_2_smiles, drug_2_idx)
 
    data["drug"].x = torch.from_numpy(morgan).float()

    df["d1"] = df["d1"].apply(lambda row: drug_2_idx[row]).to_numpy()
    df["d2"] = df["d2"].apply(lambda row: drug_2_idx[row]).to_numpy()
    
    edge_index_dict = {}
    
    for id_ in relation_id:
        edge_df = df[df.type == id_]
        d1 = edge_df["d1"]
        d2 = edge_df["d2"]
        edges = np.stack([d1, d2], axis = 0)
        edge_index_dict[str(id_)] = pyg_utils.to_undirected(
                                        torch.from_numpy(edges).long())
        
        data["drug", str(id_), "drug"].edge_index = pyg_utils.to_undirected(
                                                torch.from_numpy(edges).long())
    return data
    
def generate_label(df, drug_2_idx, relation_id, device = "cpu"):
    edge_label_index_dict = {}
    edge_label_dict = {}

    try:
        df["d1"] = df["d1"].apply(lambda row: drug_2_idx[row]).to_numpy()
        df["d2"] = df["d2"].apply(lambda row: drug_2_idx[row]).to_numpy()
    except:
        pass 
    
    for id_ in relation_id:
        edge_df = df[df.type == id_]
        d1 = edge_df["d1"]
        d2 = edge_df["d2"]
        edges = np.stack([d1, d2], axis = 0)
        pos_labels = np.zeros(edges.shape[1])
        edge_label_index_dict[str(id_)] = torch.from_numpy(edges).long().to(device)
        edge_label_dict[str(id_)] = torch.from_numpy(pos_labels).float().to(device)
    
    return edge_label_index_dict, edge_label_dict

def count_num_labels(label_dict):
    return sum(label_dict[i].shape[0] for i in label_dict.keys())
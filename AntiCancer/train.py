import argparse
import numpy as np
import pandas as pd
import torch

import torch_geometric
import torch_geometric.utils as pyg_utils
import torch_geometric.data as pyg_data
import torch_geometric.transforms as pyg_T
import networkx as nx
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import numpy as np 
from collections import defaultdict

from hetero_vgae import *
from metrics import *

def read_csv_file(filename):
    df = pd.read_csv(filename)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    return df

parser = argparse.ArgumentParser(description = "AntiCancer Link Prediction")
parser.add_argument("--random", type = int, default = 1, help = "random")
parser.add_argument("--num_random", type = int, default = 1, help = "number of randoms")
parser.add_argument("--idx_d", type = int, default = 1, help = "drug similarity type index")
parser.add_argument("--idx_c", type = int, default = 1, help = "cell similarity type index")
parser.add_argument("--dropout", type = float, default = 0.0, help = "dropout rate")
parser.add_argument("--device", type = str, default = "cuda:0", help = "training device")
parser.add_argument("--log_dir", type = str, default = "./Log/", help = "directory to log experimental results")
parser.add_argument("--chkpt_dir", type = str, default = "./Checkpoint/", help = 'checkpoint directory')

args = parser.parse_args()

dataname = "CCLE"
simD = ["Chem_SimD", "Target_SimD", "KEGG_SimD"]
simC = ["CNV_SimC", "Mutation_SimC", "Expression_SimC"]

idx_d = args.idx_d
idx_c = args.idx_c
cell_feature = f"./Data/{dataname}/Features/Tissue.csv"
simD_dir = f"./Data/{dataname}/Similarities/{simD[idx_d]}.csv"
simC_dir = f"./Data/{dataname}/Similarities/{simC[idx_c]}.csv"
logic_dir = f"./Data/{dataname}/Features/LogIC50.csv"


def load_data(filepaths):
    simD_dir = filepaths["simD_dir"]
    simC_dir = filepaths["simC_dir"]
    logic_dir = filepaths["logic_dir"]
    cell_feature = filepaths["cell_feature"]

    cell_feature = read_csv_file(cell_feature).to_numpy()
    cell_feature = np.transpose(cell_feature)
    R = read_csv_file(logic_dir).to_numpy()
    D = read_csv_file(simD_dir).to_numpy()
    C = read_csv_file(simC_dir).to_numpy()

    num_drugs = D.shape[0]
    num_cells = C.shape[0]
    
    sparse_R = sp.csr_matrix(R)
    sparse_R_t = sp.csr_matrix(R.T)

    drug_cell_edge_index, drug_cell_edge_attr = pyg_utils.from_scipy_sparse_matrix(sparse_R_t)
    cell_drug_edge_index, cell_drug_edge_attr = pyg_utils.from_scipy_sparse_matrix(sparse_R)
    
    drug_edge_index, drug_edge_attr = pyg_utils.dense_to_sparse(torch.tensor(D))
    cell_edge_index, cell_edge_attr = pyg_utils.dense_to_sparse(torch.tensor(C))

    drug_edge_attr = pyg_utils.normalized_cut(drug_edge_index, drug_edge_attr)
    cell_edge_attr = pyg_utils.normalized_cut(cell_edge_index, cell_edge_attr)

    data = pyg_data.HeteroData()

    data["drug"].x = torch.eye(num_drugs)
    data["cell"].x = torch.eye(num_cells)

    data["drug", "has_target", "cell"].edge_index = drug_cell_edge_index
    data["drug", "has_target", "cell"].edge_attr = drug_cell_edge_attr.float()
    data["drug", "has_target", "cell"].edge_label = drug_cell_edge_attr.float()

    data["cell", "get_target", "drug"].edge_index = cell_drug_edge_index
    data["cell", "get_target", "drug"].edge_attr = cell_drug_edge_attr.float()
    data["cell", "get_target", "drug"].edge_label = cell_drug_edge_attr.float()
    

    data["drug", "is_sim_d", "drug"].edge_index = drug_edge_index
    data["drug", "is_sim_d", "drug"].edge_attr = drug_edge_attr.float()
    

    data["cell", "is_sim_c", "cell"].edge_index = cell_edge_index
    data["cell", "is_sim_c", "cell"].edge_attr = cell_edge_attr.float()

    return data

if __name__=='__main__':
    if args.random == 1:
        print("Drug Similarity: ", simD[idx_d])
        print("Cell Similarity: ", simC[idx_c])
        Log = open(args.log_dir + f"{dataname}_{simD[idx_d]}_{simC[idx_c]}_logs_{args.num_random}.log", "w")
    else:
        Log = open(args.log_dir + f"{dataname}_{simD[idx_d]}_{simC[idx_c]}_logs_{args.num_random}.log", "a")
    
    filepaths = {
        "simD_dir" : simD_dir,
        "simC_dir" : simC_dir,
        "logic_dir" : logic_dir,
        "cell_feature" : cell_feature
    }

    data = load_data(filepaths)
    decoder_2_relation = {
        "linear" : ["has_target"]
    }
    relation_2_decoder = {
        "has_target" : "linear",
    }
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    input_dims = {
        "drug" : data["drug"].x.shape[1],
        "cell" : data["cell"].x.shape[1]
    }
    net = HeteroVGAE(input_dims, [16], 16, data.node_types, data.edge_types, decoder_2_relation, relation_2_decoder, 
                    latent_encoder_type = "linear", dropout =0.0).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-2)
    loss_fn = nn.MSELoss()

    edge_index_dict = data.edge_index_dict 

    transform = pyg_T.Compose([
        pyg_T.ToUndirected(),
        pyg_T.RandomLinkSplit(num_val = 0.1, num_test = 0.2, is_undirected=True, edge_types = [("drug", "has_target", "cell")]
            , rev_edge_types = [("cell", "get_target", "drug")], neg_sampling_ratio = 0.0, add_negative_train_samples =False)
    ])

    train_data, valid_data, test_data = transform(data)
    del edge_index_dict["drug", "is_sim_d", "drug"]
    del edge_index_dict["cell", "is_sim_c", "cell"]
    del edge_index_dict["cell", "get_target", "drug"]

    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    best_val_rmse = 1e9
    target = train_data.edge_label_dict["drug", "has_target", "cell"] 
    
    ratio = {
        "drug" : 0.001,
        "cell" : 0.001

    }
    
    total_results = []

    for e in range(500):
        net.train()
        optimizer.zero_grad()
        z_dict = net.encode(train_data.x_dict, train_data.edge_index_dict, train_data.edge_attr_dict)
        pred = net.decode_all_relation(z_dict, train_data.edge_label_index_dict)
        loss = 0.0
        pred = pred["has_target"]
        mse = loss_fn(pred, target)
        kl = net.kl_loss_all(ratio = ratio, reduce = "ratio")
        loss = mse + kl
        loss.backward()
        optimizer.step()
        train_rmse = mean_squared_error(target.detach().cpu().numpy(), pred.detach().cpu().numpy()) ** 0.5
    
        net.eval()
        with torch.no_grad():
            z_dict = net.encode(valid_data.x_dict, valid_data.edge_index_dict, valid_data.edge_attr_dict)
            pred = net.decode_all_relation(z_dict, valid_data.edge_label_index_dict)
            valid_target = valid_data.edge_label_dict["drug", "has_target", "cell"]
            pred = pred["has_target"]
            valid_rmse = mean_squared_error(valid_target.detach().cpu().numpy(), pred.detach().cpu().numpy()) ** 0.5 

            if best_val_rmse > valid_rmse:
                best_val_rmse = valid_rmse
                torch.save(net.state_dict(), args.chkpt_dir + f"{dataname}_{simD[idx_d]}_{simC[idx_c]}_{args.num_random}.pt")
  
    
    net.load_state_dict(torch.load(args.chkpt_dir + f"{dataname}_{simD[idx_d]}_{simC[idx_c]}_{args.num_random}.pt"))
    net.eval()
    with torch.no_grad():
        z_dict = net.encode(test_data.x_dict, test_data.edge_index_dict, test_data.edge_attr_dict)
        pred = net.decode_all_relation(z_dict, test_data.edge_label_index_dict)
        test_target = test_data.edge_label_dict["drug", "has_target", "cell"] 
        pred = pred["has_target"] 
        results = evaluate(test_target.detach().cpu().numpy(), pred.detach().cpu().numpy())
    total_results.append(results)

rmse = np.average([results["rmse"] for results in total_results]) 
mse = np.average([results["mse"] for results in total_results])
r2 = np.average([results["r2"] for results in total_results]) 
pearson = np.average([results["pear"] for results in total_results])
fit = np.average([results["fit"] for results in total_results]) 

Log.write(f"{mse} | {rmse} | {r2} | {pearson} | {fit}\n")
Log.close()
print(f"---------- Finish Random {args.random}-----------\n")

if args.random == args.num_random:
    metrics = defaultdict(list)
    with open(args.log_dir + f"{dataname}_{simD[idx_d]}_{simC[idx_c]}_logs_{args.num_random}.log", "r") as f:
        lines = f.readlines()

        for line in lines: 
            mse, rmse, r2, pcc, fit = line.strip("\n").split("|")
            metrics["mse"].append(float(mse))
            metrics["rmse"].append(float(rmse))
            metrics["r2"].append(float(r2))
            metrics["pcc"].append(float(pcc))
            metrics["fit"].append(float(fit))
        

    metrics["mse"] = np.array(metrics["mse"])
    metrics["rmse"] = np.array(metrics["rmse"])
    metrics["r2"] = np.array(metrics["r2"])
    metrics["pcc"] = np.array(metrics["pcc"])
    metrics["fit"] = np.array(metrics["fit"])


    Log = open(args.log_dir + f"{dataname}_{simD[idx_d]}_{simC[idx_c]}_logs_{args.num_random}.log", "a")
    Log.write("\n")
    Log.write("| Mean MSE: {} | Std MSE: {}\n".format(metrics["mse"].mean(), metrics["mse"].std()))
    Log.write("| Mean RMSE: {} | Std RMSE: {}\n".format(metrics["rmse"].mean(), metrics["rmse"].std()))
    Log.write("| Mean R2: {} | Std R2: {}\n".format(metrics["r2"].mean(), metrics["r2"].std()))
    Log.write("| Mean PCC: {} | Std PCC: {}\n".format(metrics["pcc"].mean(), metrics["pcc"].std()))
    Log.write("| Mean Fit: {} | Std Fit: {}\n".format(metrics["fit"].mean(), metrics["fit"].std()))
    Log.close()
    print(f"---------- Finish All {args.num_random} Random Seed -----------\n")
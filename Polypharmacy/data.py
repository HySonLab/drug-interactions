import torch_geometric.utils as pyg_utils
import torch_geometric.data as pyg_data

import json

from utils import *

ppi_path = "./Data/bio-decagon-ppi/bio-decagon-ppi.csv"
combo_side_effect_path = "./Data/bio-decagon-combo/bio-decagon-combo.csv"
drug_gene_path = "./Data/bio-decagon-targets/bio-decagon-targets.csv"
drug_smile_path = "./Data/drug_smile.json"
mono_side_effect_path = "./Data/bio-decagon-mono/bio-decagon-mono.csv"

            
def load_data(return_augment =False):
    """ protein - protein """
    net, gene_2_idx = load_ppi(ppi_path)

    gene_edge_list = list(net.edges.data())
    gene_edge_index = [[gene_2_idx[gene_edge_list[i][0]], gene_2_idx[gene_edge_list[i][1]]] for 
                    i in range(len(gene_edge_list))]
    gene_edge_index = torch.tensor(gene_edge_index).long().T

    """ drug - drug """
    combo_2_se, se_2_combo, se_2_name, combo_2_stitch, stitch_2_idx = load_combo_side_effect(combo_side_effect_path)
    edge_index_dict = defaultdict(list)

    drug_2_idx, edge_index_dict = convert_combo_side_effect_to_edge_index_list(
                                    se_2_combo, combo_2_stitch, stitch_2_idx)

    print("Number of side effects in consideration: ", len(edge_index_dict))

    """ drug - protein """
    drug_2_gene = load_targets(drug_gene_path)
    drug_gene_edge_index = []
    for stitch, genes in drug_2_gene.items():
        for gene in genes:
            try:
                drug_gene_edge_index.append([stitch_2_idx[stitch], gene_2_idx[gene]])
            except:
                pass

    drug_gene_edge_index = torch.tensor(drug_gene_edge_index).long().T

    index = torch.LongTensor([1,0])
    gene_drug_edge_index = torch.zeros_like(drug_gene_edge_index)
    gene_drug_edge_index[index] = drug_gene_edge_index

    edge_index_dict[("drug", "has_target", "gene")] = drug_gene_edge_index
    edge_index_dict[("gene", "get_target", "drug")] = gene_drug_edge_index
    edge_index_dict[("gene", "interact", "gene")] = gene_edge_index

    data = pyg_data.HeteroData()

    data["gene"].x = torch.eye(len(gene_2_idx))
    data["drug"].x = torch.eye(len(stitch_2_idx))
    
    if return_augment:
        with open(drug_smile_path, "r") as f:
            stitch_2_smile = json.load(f)
        data["drug"].augment = torch.tensor(generate_morgan_fingerprint(stitch_2_smile, stitch_2_idx))

    for (src, relation, dst) in edge_index_dict.keys():
        data[src, relation, dst].edge_index = edge_index_dict[(src, relation, dst)]

    for edge_type in data.edge_types:
        data[edge_type].edge_index = pyg_utils.sort_edge_index(data[edge_type].edge_index)
    
    return data
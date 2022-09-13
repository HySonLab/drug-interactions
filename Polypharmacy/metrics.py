import torch
from operator import itemgetter
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

def concat_all(item_dict):
    return torch.cat([item_dict[relation] for relation in item_dict.keys()], dim = -1)

def cal_roc_auc_score(pred, label, edge_types):
    all_label = concat_all(label)
    all_pred = concat_all(pred)
    return roc_auc_score(all_label, all_pred)

def cal_acc_score(pred, label, edge_types):
    all_label = concat_all(label)
    all_pred = concat_all(pred)
    return accuracy_score(all_label, all_pred)

def cal_average_precision_score(pred, label):
    all_label = concat_all(label)
    all_pred = concat_all(pred)
    return average_precision_score(pred, label)

def cal_roc_auc_score_per_side_effect(preds, labels, edge_types):
    total_roc_auc = {}

    for src, relation, dst in edge_types:
        if relation not in ["has_target", "get_target", "interact"]:
            score = roc_auc_score(labels[relation], preds[relation])
            total_roc_auc[relation] = score

    return sum(total_roc_auc.values()) / len(total_roc_auc), total_roc_auc

def cal_average_precision_score_per_side_effect(preds, labels, edge_types):
    total_prec = {}

    for src, relation, dst in edge_types:
        if relation not in ["has_target", "get_target", "interact"]:
            score = average_precision_score(labels[relation], preds[relation])
            total_prec[relation] = score 
    
    return sum(total_prec.values()) / len(total_prec), total_prec

def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def cal_apk(preds, labels, edge_types, k = 50):
    
    total_apk = {}

    for src, relation, dst in edge_types:
        if relation in ["has_target", "get_target", "interact"]:
            continue

        actual = []
        predicted = []
        edge_ind = 0

        for idx, score in enumerate(preds[relation]):
            if labels[relation][idx] == 1:
                actual.append(idx)
            predicted.append((score, idx))
        predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]
        total_apk[relation] = apk(actual, predicted, k = k)
    
    return sum(total_apk.values()) / len(total_apk), total_apk


    

import torch
import pdb


def compute_pn_mask(label):
    pos_mask = torch.zeros([label.shape[0], label.shape[0]]).cuda()
    neg_mask = torch.ones([label.shape[0], label.shape[0]]).cuda()
    for i in range(label.shape[0]):
        pos_mask[i, :] = (label[i] == label)
    neg_mask = neg_mask - pos_mask
    return pos_mask, neg_mask

def comput_hard_triplet(feat, label, margin=0.2):
    pos_mask, neg_mask = compute_pn_mask(label)

    feat_norm = (feat*feat).sum(dim=1).view(feat.shape[0], 1)
    feat = feat / feat_norm
    
    feat_row = feat.view(feat.shape[0], 1, -1)  
    feat_col = feat.view(1, feat.shape[0], -1)
    feat_row = feat_row.repeat(1, feat.shape[0], 1) 
    feat_col = feat_col.repeat(feat.shape[0], 1, 1)  
    feat_dist = (feat_row - feat_col) * (feat_row - feat_col) 
    feat_dist = feat_dist.sum(dim=2)
    
    pos_dist = feat_dist * pos_mask
    neg_dist = feat_dist * neg_mask
    pos_hard = pos_dist.max(dim=1)[0]
    neg_hard = (neg_dist + 1).min(dim=1)[0] - 1
    
    y_predict = torch.topk(feat_dist + torch.eye(feat.shape[0]).cuda(), dim=1, k=1, largest=False)[1]
    y_predict = label[y_predict]
    acc = (y_predict.view(-1)==label).float().mean()
    loss = torch.clamp(pos_hard - neg_hard + margin, min=0).mean()
    
    return loss, acc
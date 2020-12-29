import torch
import pdb


def compute_pn_mask(label):
    pos_mask = torch.zeros([label.shape[0], label.shape[0]]).cuda()
    neg_mask = torch.ones([label.shape[0], label.shape[0]]).cuda()
    for i in range(label.shape[0]):
        pos_mask[i, :] = (label[i] == label)
    neg_mask = neg_mask - pos_mask
    return pos_mask, neg_mask

def compute_pn_mask_img(label):
    pos_mask = torch.zeros([label.shape[0] // 5, label.shape[0]]).cuda()
    neg_mask = torch.ones([label.shape[0] // 5, label.shape[0]]).cuda()
    for i in range(label.shape[0] // 5):
        pos_mask[i, :] = ((i + 2) == label)
    neg_mask = neg_mask - pos_mask
    return pos_mask, neg_mask

def count_value(preds):
    preds = preds.transpose(1, 0)
    idx = [i + 2 for i in range(int(preds.shape[1] / 5))]
    count = torch.zeros(preds.shape).cuda()
    for label in idx:
        tmp = (preds == label).float()
        cnt = torch.sum(tmp, dim=0)
        temp = tmp * cnt
        count += temp
    return count.transpose(1, 0)

def compute_hard_triplet(feat, label, epoch, margin=0.8):
    pos_mask, neg_mask = compute_pn_mask(label)

    feat_norm = abs(feat).sum(dim=1).view(feat.shape[0], 1)
    feat = feat / feat_norm
    
    feat_row = feat.view(feat.shape[0], 1, -1)  
    feat_col = feat.view(1, feat.shape[0], -1)
    feat_row = feat_row.repeat(1, feat.shape[0], 1) 
    feat_col = feat_col.repeat(feat.shape[0], 1, 1)  
    #feat_dist = (feat_row - feat_col) * (feat_row - feat_col) 
    feat_dist = abs(feat_row - feat_col)
    feat_dist = feat_dist.sum(dim=2)
    
    pos_dist = feat_dist * pos_mask
    neg_dist = feat_dist * neg_mask

    if epoch < 5:
        pos_hard = pos_dist.sum(dim=1) / pos_mask.sum(dim=1)
        neg_hard = neg_dist.sum(dim=1) / neg_mask.sum(dim=1)
    elif epoch < 10:
        pos_hard = pos_dist.max(dim=1)[0]
        neg_hard = neg_dist.sum(dim=1) / neg_mask.sum(dim=1)
    else:
        pos_hard = pos_dist.max(dim=1)[0]
        neg_hard = (neg_dist + (1 - neg_mask.cuda())).min(dim=1)[0]
    # pos_hard = pos_dist.max(dim=1)[0]
    # neg_hard = (neg_dist + (1 - neg_mask.cuda())).min(dim=1)[0]
    mask = (neg_hard > pos_hard).float()
    
    y_predict = torch.topk(feat_dist + torch.eye(feat.shape[0]).cuda(), dim=1, k=1, largest=False)[1]
    y_predict = label[y_predict]

    acc = (y_predict.view(-1) == label).float().mean()
    loss = (torch.clamp(pos_hard - neg_hard + margin, min=0)).mean() - mask.mean()
    # idx = torch.topk(feat_dist + torch.eye(feat.shape[0]).cuda(), dim=1, k=4, largest=False)[1]
    # # pdb.set_trace()
    # predictions = label.index_select(0, idx.view(-1))  # [batch_size * 5 * 4]
    # predictions = predictions.reshape(-1, 4).flip(1)  # [batch_size * 5, 4]
    # count = count_value(predictions)
    # y_idx = torch.argmax(count, dim=1).cuda()
    # id = torch.LongTensor([[i for i in range(predictions.shape[0])]]).cuda()
    # y_predict = predictions.index_select(1, y_idx.view(-1)).gather(0, id)

    # acc = (y_predict.view(-1) == label).float().mean()
    # loss = (torch.clamp(pos_hard - neg_hard + margin, min=0)).mean() - 10 * mask.mean()

    return loss, acc

def compute_img_loss(text_feat, label, img_feat, epoch, margin=0.8):
    pos_mask, neg_mask = compute_pn_mask_img(label)

    feat_norm = abs(text_feat).sum(dim=1).view(text_feat.shape[0], 1)
    text_feat = text_feat / feat_norm
    feat_norm = abs(img_feat).sum(dim=1).view(img_feat.shape[0], 1)
    img_feat = img_feat / feat_norm

    feat_row = img_feat.view(img_feat.shape[0], 1, -1)  
    feat_col = text_feat.view(1, text_feat.shape[0], -1)
    feat_row = feat_row.repeat(1, text_feat.shape[0], 1) 
    feat_col = feat_col.repeat(img_feat.shape[0], 1, 1)  

    feat_dist = abs(feat_row - feat_col)
    feat_dist = feat_dist.sum(dim=2)

    pos_dist = feat_dist * pos_mask
    neg_dist = feat_dist * neg_mask

    if epoch < 5:
        pos_hard = pos_dist.sum(dim=1) / pos_mask.sum(dim=1)
        neg_hard = neg_dist.sum(dim=1) / neg_mask.sum(dim=1)
    elif epoch < 10:
        pos_hard = pos_dist.max(dim=1)[0]
        neg_hard = neg_dist.sum(dim=1) / neg_mask.sum(dim=1)
    else:
        pos_hard = pos_dist.max(dim=1)[0]
        neg_hard = (neg_dist + (1 - neg_mask.cuda())).min(dim=1)[0]

    mask = (neg_hard > pos_hard).float()
    y_predict = torch.topk(feat_dist, dim=1, k=1, largest=False)[1]
    y_predict = label[y_predict]

    img_label = torch.linspace(2, 1 + img_feat.shape[0], img_feat.shape[0]).cuda().long()
    acc = (y_predict.view(-1) == img_label).float().mean()
    loss = (torch.clamp(pos_hard - neg_hard + margin, min=0)).mean() - mask.mean()

    return loss, acc

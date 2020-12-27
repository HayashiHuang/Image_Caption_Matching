from text_preprocess import read_text_data
from dataset import TrainDataset, ImgDataset, CaptionDataset
import torch
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
from torchvision import transforms
import os
import pdb

def train_collate_fn(data):
    img_data = [img_cap[0] for img_cap in data]
    img_data = torch.stack(img_data)
    cap_data = [img_cap[1] for img_cap in data]
    #cap_data.sort(key=lambda x: x.shape[0])
    cap_data = rnn_utils.pad_sequence(cap_data, batch_first=True, padding_value=1)
    return img_data, cap_data.transpose(1, 2)

def test_collate_fn(data):
    if len(data[0]) == 2:
        cap_data = [cap[0] for cap in data]
        label_data = [cap[1] for cap in data]
        #cap_data.sort(key=lambda x: len(x))
        cap_data = rnn_utils.pad_sequence(cap_data, batch_first=True, padding_value=1)
        cap_data = cap_data.view(cap_data.shape[0], cap_data.shape[1])
        return cap_data, torch.tensor(label_data)
    else:
        data = [cap for cap in data]
        data.sort(key=lambda x: len(x))
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=1)
        data = data.view(data.shape[0], data.shape[1])
        return data

def load_dataset(args):

    img_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.RandomCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_caption, validation_caption, test_caption, train_id, validation_id, val_img_id, test_img_id, img_idx, vocab_table = read_text_data(args)
    
    train_dataset = TrainDataset(args.img_dir, vocab_table, img_idx, train_caption, train_id, img_transform)
    val_img_dataset = ImgDataset(args.img_dir, val_img_id, is_val=True, transform=img_transform, img_idx=img_idx)
    test_img_dataset = ImgDataset(args.img_dir, test_img_id, transform=img_transform)
    val_caption_dataset = CaptionDataset(vocab_table, validation_caption, is_val=True, label=validation_id)
    test_caption_dataset = CaptionDataset(vocab_table, test_caption)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, collate_fn=train_collate_fn)
    val_img_loader = DataLoader(val_img_dataset, batch_size=args.batch_size, num_workers=4)
    test_img_loader = DataLoader(test_img_dataset, batch_size=args.batch_size, num_workers=4)
    val_caption_loader = DataLoader(val_caption_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=test_collate_fn)
    test_caption_loader = DataLoader(test_caption_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=test_collate_fn)

    return train_loader, val_img_loader, test_img_loader, val_caption_loader, test_caption_loader, vocab_table
    

from torch.utils.data import Dataset
from PIL import Image
import os
import pdb

class TrainDataset(Dataset):
    def __init__(self, img_dir, vocab_table, img_idx, caption, label, transform=None):
        self.img_dir = img_dir  
        self.vocab_table = vocab_table
        self.img_idx = img_idx
        self.caption = caption
        self.label = label
        self.transform = transform 

    def __getitem__(self, index):
        caption = [self.vocab_table.preprocess(cap) for cap in self.caption[index]]
        caption = self.vocab_table.pad(caption)
        caption = self.vocab_table.numericalize(caption)
        label = self.label[index]

        img_path = os.path.join(self.img_dir, self.img_idx[label] + '.jpg')
        img = Image.open(img_path)
        
        if self.transform is not None:
            img = self.transform(img) 
        return img, caption

    def __len__(self):
        return len(self.caption)

class ImgDataset(Dataset):
    def __init__(self, img_dir, img_list, is_val=False, transform=None, img_idx=None):
        self.img_dir = img_dir 
        self.img_list = img_list
        self.is_val = is_val
        self.transform = transform
        if is_val:
            self.img_idx = img_idx 

    def __getitem__(self, index):
        if self.is_val:
            img_path = os.path.join(self.img_dir, self.img_idx[self.img_list[index]] + '.jpg')
            img = Image.open(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return img, self.img_list[index]
        else:
            img_path = os.path.join(self.img_dir, self.img_list[index] + '.jpg')
            img = Image.open(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return img

    def __len__(self):
        return len(self.img_list)

class CaptionDataset(Dataset):
    def __init__(self, vocab_table, caption, is_val=False, label=None):
        self.vocab_table = vocab_table
        self.caption = caption
        self.is_val = is_val
        if is_val:
            self.label = label

    def __getitem__(self, index):
        caption = self.vocab_table.preprocess(self.caption[index])
        caption = self.vocab_table.numericalize([caption])
        
        if self.is_val:
            label = self.label[index]
            return caption, label
        else:
            return caption

    def __len__(self):
        return len(self.caption)

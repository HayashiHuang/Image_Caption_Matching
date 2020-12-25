import torch
import torchtext
import torch.utils.data as Data
from torchtext.vocab import Vectors
from nltk.stem import WordNetLemmatizer
import os
import re
import random
import pdb

def flatten(list_in):
    list_out = []
    [list_out.extend(i) for i in list_in]
    return list_out

def tokenizer(sent):
    
    language_remove = re.compile("[^\u4e00-\u9fa5^.^a-z^A-Z^0-9]")
    punctuation_remove = u'[、：，？！。·；……（）『』《》【】～!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    sent = re.sub(r'ldquo', " ", sent)
    sent = re.sub(r'hellip', " ", sent)
    sent = re.sub(r'rdquo', " ", sent)
    sent = re.sub(r'yen', " ", sent)
    sent = re.sub(r'⑦', "7", sent)
    sent = re.sub(r'(， ){2,}', " ", sent)
    sent = re.sub(r'(！ ){2,}', " ", sent)
    sent = re.sub(r'(？ ){2,}', " ", sent)
    sent = re.sub(r'(。 ){2,}', " ", sent)
    sent = re.sub(punctuation_remove, " ", sent)
    sent = re.sub(language_remove, " ", sent)
    sent = sent.lower().split()
    
    max_len = 100
    if(len(sent) > max_len):
        sent = sent[:max_len]

    wnl = WordNetLemmatizer()
    sent = [wnl.lemmatize(s) for s in sent]

    return sent

def read_text_data(args):
    train_data = [line for line in open(os.path.join(args.text_dir, "Train_Caption.txt"))]
    test_caption = [line[:-1] for line in open(os.path.join(args.text_dir, "Test_CaptionPool.txt"))]
    test_img_id = [line[:-1] for line in open(os.path.join(args.text_dir, "Test_ImageName.txt"))]

    train_data = train_data[1:]
    train_dict = [line[:-1] for line in open(os.path.join(args.text_dir, "Train_ImageName.txt"))]
    train_dict = dict(zip([i for i in range(len(train_dict))], train_dict))
    train_data = [train_data[5 * i: 5 * i + 5] for i in range(len(train_dict))]
    
    text_id = [i for i in range(len(train_dict))]
    
    random.shuffle(text_id)
    train_id = text_id[:int(args.split_ratio * len(train_dict))]
    train_caption = [[data.split("|")[-1][:-1] for data in train_data[idx]] for idx in train_id]
    
    validation_id = text_id[int(args.split_ratio * len(train_dict)):]
    val_img_id = validation_id

    validation_caption = flatten([[data.split("|")[-1][:-1] for data in train_data[idx]] for idx in validation_id])
    validation_id = flatten([[idx] * 5 for idx in validation_id])

    text = torchtext.data.Field(sequential=True, tokenize=tokenizer, lower=True)
    vectors = Vectors(name='glove.840B.300d.txt', cache='./data')
    text.build_vocab([tokenizer(word) for word in flatten(train_caption)], max_size=args.max_size, min_freq=args.min_freq, vectors=vectors, unk_init=torch.Tensor.normal_)
    
    return train_caption, validation_caption, test_caption, train_id, validation_id, val_img_id, test_img_id, train_dict, text


